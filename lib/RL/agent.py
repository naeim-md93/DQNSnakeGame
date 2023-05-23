import math
import torch
import random
import numpy as np
from collections import deque

from .agent_objects.model import QModel
from .agent_objects.memory_pool import MemoryPool

import env


class QAgent:
    def __init__(self, device: torch.device, checkpoints_path: str) -> None:
        self.device = device
        self.memory_pool = MemoryPool(
            size=env.MAX_MEMORY,
            proportions=[
                env.NMP1_PROPORTION,
                env.NMP2_PROPORTION,
                env.PMP2_PROPORTION,
                env.PMP1_PROPORTION
            ]
        )
        self.model = QModel(num_classes=len(env.SNAKE_ACTIONS), save_path=checkpoints_path).to(device=device)

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=env.LEARNING_RATE)
        self.criterion = torch.nn.MSELoss(reduction='none')

    def one_hot_state(self, state):
        state = np.stack((
            state == 0,
            state == 1,
            state == 2,
            state == 3,
            state == 4
        ), axis=1)
        return state

    def get_agent_action(
            self,
            state2ds: deque[np.ndarray],
            state1ds: deque[np.ndarray],
            legal_moves: np.ndarray,
            epsilon: float
    ) -> tuple[np.ndarray, float]:

        if env.DEBUG:
            print(f'{"#" * 10} GET AGENT ACTION {"#" * 10}')
            print(f'{epsilon=}')
            print(f'state2ds: {len(state2ds)} * {state2ds[0].shape}')
            print(f'state1ds: {len(state1ds)} * {state1ds[0].shape}')
            print(f'legal_moves: {legal_moves=}')

        pred = None
        random_value = np.random.random()
        action = np.zeros(shape=(len(env.SNAKE_ACTIONS), ), dtype=np.uint8)

        # Random action
        if random_value < epsilon:
            random_action = np.random.random(size=(len(env.SNAKE_ACTIONS), )) * legal_moves
            action[np.argmax(a=random_action, axis=0)] = 1

        # Model action
        else:
            state2ds = [np.expand_dims(a=s, axis=0) for s in state2ds]  # [(1, 14, 14), ..]
            state2ds = [self.one_hot_state(state=s) for s in state2ds]  # [(1, 5, 14, 14), ..]
            state2ds = np.concatenate(state2ds, axis=0)  # (t, 5, 14, 14)
            state2ds = torch.tensor(data=state2ds, dtype=torch.float).unsqueeze(0)  # (b, t, 5, 14, 14)
            state2ds = state2ds.to(device=self.device)

            state1ds = np.stack(arrays=state1ds, axis=0)  # (t, 12)
            state1ds = torch.tensor(data=state1ds, dtype=torch.float).unsqueeze(0)  # (b, t, 12)
            state1ds = state1ds.to(device=self.device)

            self.model.eval()

            with torch.no_grad():
                pred = self.model(state2ds, state1ds).squeeze(0).detach().cpu().numpy()

            pred = np.where(legal_moves == 1, pred, -np.inf)
            action[np.argmax(a=pred, axis=0)] = 1

        if env.DEBUG:
            print(f'state2ds: {len(state2ds)} * {state2ds[0].shape}')
            print(f'state1ds: {len(state1ds)} * {state1ds[0].shape}')
            print(f'{random_value=}')
            print(f'{pred=}')
            print(f'action: {action.shape} {action=}')
            input(f'{"#" * 10} GET AGENT ACTION {"#" * 10}')

        return action, random_value

    def remember(self, xp: tuple, reward: float) -> list[int, int, int, int]:

        if -env.MEMORY_THRESHOLD >= reward:
            self.memory_pool.store_in_NMP1(xp=xp)

        elif -env.MEMORY_THRESHOLD < reward <= 0:
            self.memory_pool.store_in_NMP2(xp=xp)

        elif 0 < reward <= env.MEMORY_THRESHOLD:
            self.memory_pool.store_in_PMP2(xp=xp)

        elif reward > env.MEMORY_THRESHOLD:
            self.memory_pool.store_in_PMP1(xp=xp)
        else:
            raise ValueError('UNKNOWN')

        sizes = [
            len(self.memory_pool.NMP1),
            len(self.memory_pool.NMP2),
            len(self.memory_pool.PMP2),
            len(self.memory_pool.PMP1),
        ]

        if env.DEBUG:
            print(f'{"#" * 10} REMEMBERING {"#" * 10}')
            print(f'{reward=}')
            print(f'{sizes=}')
            input(f'{"#" * 10} REMEMBERING {"#" * 10}')

        return sizes

    def train_model(self, xps: tuple, gamma: float):

        self.model.train()

        state2d0s = torch.tensor(data=xps[0], dtype=torch.float).to(device=self.device)
        state1d0s = torch.tensor(data=xps[1], dtype=torch.float).to(device=self.device)
        legal_moves0 = xps[2]
        actions = xps[3]
        rewards = xps[4]
        game_overs = xps[5]
        state2d1s = torch.tensor(data=xps[6], dtype=torch.float).to(device=self.device)
        state1d1s = torch.tensor(data=xps[7], dtype=torch.float).to(device=self.device)
        legal_moves1 = xps[8]

        state1s_pred = self.model(state2d1s, state1d1s).detach().cpu().numpy()
        state1s_pred = np.where(legal_moves1 == 1, state1s_pred, -np.inf)

        state0s_pred = self.model(state2d0s, state1d0s)

        target = state0s_pred.clone().detach().cpu().numpy()
        target_non_action = target * (1 - actions)

        tmp = gamma * np.amax(a=state1s_pred, axis=1, keepdims=True) * (1 - game_overs)
        target_action = rewards + tmp
        target_action = target_action * actions
        target = target_action + target_non_action
        target = torch.tensor(data=target, dtype=torch.float).to(device=self.device)
        loss = torch.nanmean(input=self.criterion(input=state0s_pred, target=target), dim=0)

        self.optimizer.zero_grad()
        loss.sum().backward()
        self.optimizer.step()

        if env.DEBUG:
            print(f'state2d0s: {state2d0s.shape}')
            print(f'state1d0s: {state1d0s.shape}')
            print(f'legal_moves0: {legal_moves0.shape}')
            print(f'actions: {actions.shape}')
            print(f'rewards: {rewards.shape}')
            print(f'game_overs: {game_overs.shape}')
            print(f'state2d1s: {state2d1s.shape}')
            print(f'state1d1s: {state1d1s.shape}')
            print(f'legal_moves1: {legal_moves1.shape}')
            print(f'{state0s_pred=}')
            print(f'{state1s_pred=}')
            print(f'{target_non_action=}')
            print(f'{target_action=}')
            print(f'{target=}')
            print(f'{loss=}')

        return loss.detach().cpu().numpy()

    def train_short_memory(self, xp: tuple):

        if env.DEBUG:
            print(f'{"#" * 10} TRAIN SHORT MEMORY {"#" * 10}')
            print(f'{env.SHORT_MEMORY_GAMMA=}')
            for i in range(len(xp)):
                print(xp[i].shape)

        state2d0 = xp[0]  # (t, 14, 14)
        state2d0 = [np.expand_dims(a=state2d0[t], axis=0) for t in range(state2d0.shape[0])]  # [(1, 14, 14), ...]
        state2d0 = [self.one_hot_state(state=x) for x in state2d0]  # [(1, 5, 14, 14), ...]
        state2d0 = np.concatenate(state2d0, axis=0)  # (t, 5, 14, 14)
        state2d0 = np.expand_dims(a=state2d0, axis=0)  # (1, t, 5, 14, 14)

        state2d1 = xp[6]  # (t + 1, 14, 14)
        state2d1 = [np.expand_dims(a=state2d1[t], axis=0) for t in range(state2d1.shape[0])]  # [(1, 14, 14), ...]
        state2d1 = [self.one_hot_state(state=x) for x in state2d1]  # [(1, 5, 14, 14), ...]
        state2d1 = np.concatenate(state2d1, axis=0)  # (t + 1, 5, 14, 14)
        state2d1 = np.expand_dims(a=state2d1, axis=0)  # (1, t, 5, 14, 14)

        xp = (
            state2d0,  # State2d0 (1, t, 5, 14, 14)
            np.expand_dims(a=xp[1], axis=0),  # (b, t, 12)
            xp[2],  # Legal Moves 0 (1, 4)
            xp[3],  # Action (1, 4)
            xp[4],  # Reward (1, 1)
            xp[5],  # Game Over (1, 1)
            state2d1,  # State1 (1, t + 1, 14, 14)
            np.expand_dims(a=xp[7], axis=0),  # (b, t + 1, 12)
            xp[8],  # Legal Moves 1 (1, 4)
        )

        if env.DEBUG:
            for i in range(len(xp)):
                print(xp[i].shape)

        losses = self.train_model(xps=xp, gamma=env.SHORT_MEMORY_GAMMA)

        if env.DEBUG:
            input(f'{"#" * 10} TRAIN SHORT MEMORY {"#" * 10}')

        return losses

    def train_long_memory(self, eta):

        loss = (0, 0, 0, 0)

        if len(self.memory_pool.NMP1) and len(self.memory_pool.NMP2) and len(self.memory_pool.PMP2) and len(self.memory_pool.PMP1):

            MP1_samples = min(
                math.ceil((eta * env.BATCH_SIZE) / 2),
                 len(self.memory_pool.PMP1),
                 len(self.memory_pool.NMP1)
            )
            MP2_samples = min(
                math.floor(((1 - eta) * env.BATCH_SIZE) / 2),
                 len(self.memory_pool.PMP2),
                 len(self.memory_pool.NMP2)
            )

            if len(self.memory_pool.NMP1) > MP1_samples:
                miniNMP1 = random.sample(population=self.memory_pool.NMP1, k=MP1_samples)
            else:
                miniNMP1 = self.memory_pool.NMP1

            if len(self.memory_pool.NMP2) > MP2_samples:
                miniNMP2 = random.sample(population=self.memory_pool.NMP2, k=MP2_samples)
            else:
                miniNMP2 = self.memory_pool.NMP2

            if len(self.memory_pool.PMP2) > MP2_samples:
                miniPMP2 = random.sample(population=self.memory_pool.PMP2, k=MP2_samples)
            else:
                miniPMP2 = self.memory_pool.PMP2

            if len(self.memory_pool.PMP1) > MP1_samples:
                miniPMP1 = random.sample(population=self.memory_pool.PMP1, k=MP1_samples)
            else:
                miniPMP1 = self.memory_pool.PMP1

            mini_sample = list(miniNMP1) + list(miniNMP2) + list(miniPMP2) + list(miniPMP1)
            random.shuffle(x=mini_sample)

            state2d0 = [m[0] for m in mini_sample]  # [(t, 14, 14), ...]
            state2d0 = [np.expand_dims(a=x, axis=1) for x in state2d0]  # [(t, 1, 14, 14), ...]
            state2d0 = [np.concatenate([self.one_hot_state(state=s[t]) for t in s.shape[0]], axis=0) for s in state2d0]  # [(t, 5, 14, 14), ...]
            state2d0 = np.stack(arrays=state2d0, axis=0)  # (b, t, 5, 14, 14)

            state1d0 = np.stack(arrays=[m[1] for m in mini_sample], axis=0)  # (b, t, 12)

            legal_moves0 = np.concatenate([m[2] for m in mini_sample], axis=0)
            actions = np.concatenate([m[3] for m in mini_sample], axis=0)
            rewards = np.concatenate([m[4] for m in mini_sample], axis=0)
            game_overs = np.concatenate([m[5] for m in mini_sample], axis=0)

            state2d1 = [m[6] for m in mini_sample]  # [(t + 1, 14, 14), ...]
            state2d1 = [np.expand_dims(a=x, axis=1) for x in state2d1]  # [(t + 1, 1, 14, 14), ...]
            state2d1 = [np.concatenate([self.one_hot_state(state=s[t]) for t in s.shape[0]], axis=0) for s in state2d1]  # [(t + 1, 5, 14, 14), ...]
            state2d1 = np.stack(arrays=state2d1, axis=0)  # (b, t + 1, 5, 14, 14)

            state1d1 = np.stack(arrays=[m[7] for m in mini_sample], axis=0)  # (b, t + 1, 12)

            legal_moves1 = np.concatenate([m[8] for m in mini_sample], axis=0)

            xps = (state2d0, state1d0, legal_moves0, actions, rewards, game_overs, state2d1, state1d1, legal_moves1)

            if env.DEBUG:
                print(f'{"#" * 10} TRAIN LONG TERM MEMORY {"#" * 10}')
                print(f'{eta=}')
                print(f'{len(self.memory_pool.NMP1)=}, {len(self.memory_pool.NMP2)=}, {len(self.memory_pool.PMP2)=}, {len(self.memory_pool.PMP1)=}')
                print(f'{MP1_samples=}, {MP2_samples=}')
                print(f'{len(miniNMP1)=}, {len(miniNMP2)=}, {len(miniPMP2)=}, {len(miniPMP1)=}')
                print(f'{len(mini_sample)=}')

            loss = self.train_model(xps=xps, gamma=env.LONG_MEMORY_GAMMA)

            if env.DEBUG:
                input(f'{"#" * 10} TRAIN LONG TERM MEMORY {"#" * 10}')

        return loss
