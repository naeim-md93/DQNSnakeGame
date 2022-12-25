import torch
import numpy as np
import CONSTANTS
import torchvision.transforms as T

from src.RL.agents.utils.model import Model


class Agent:
    def __init__(self, args):
        self.device = args.device
        self.trans = T.ToTensor()
        self.model = Model(num_classes=args.num_snake_actions, save_path=args.models_path)

    def get_agent_action(self, **kwargs):
        state = kwargs['state0']  # (14, 14)
        legal_moves = kwargs['legal_moves0']  # (1, 4)
        epsilon = kwargs['epsilon']

        random_value = np.random.random()
        action = np.zeros(shape=(len(CONSTANTS.BOARD_DIRECTIONS), ), dtype=np.uint8)

        # If random action
        if random_value < epsilon:
            random_action = np.random.random(size=(len(CONSTANTS.BOARD_DIRECTIONS), )) * legal_moves
            action[np.argmax(a=random_action, axis=0)] = 1

        else:
            state = self.trans(state).unsqueeze(0)
            state = state.to(device=self.device)

            self.model.eval()
            with torch.no_grad():
                pred = self.model(state)
            pred = pred.detach().cpu().numpy()
            pred = np.where(legal_moves == 1, pred, -np.inf)
            action[np.argmax(a=pred, axis=0)] = 1

        return action
