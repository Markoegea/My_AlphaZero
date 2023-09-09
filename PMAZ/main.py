#External libraries
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
#Internal libraries
from game import *
from model import *
from agent import MCTS, AlphaZero
#Runtime variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args = {
    'C': 2,
    'num_searches': 600,
    'num_iterations':8,
    'num_selfPlay_iterations':500,
    'num_parallel_games':100,
    'num_epochs':4,
    'batch_size':32,
    'num_resBlocks': 4,
    'num_hidden' : 64,
    'temperature':1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3,
}
#Init game TicTacToe or ConnectFour
game = ConnectFour()
player = 1
state = game.get_initial_state()

def play_alone(game, player):
    state = game.get_initial_state()
    while True:
       print(state)
       if player == 1:
           valid_moves = game.get_valid_moves(state)
           print("valid_moves", [i for i in range(game.action_size) if valid_moves[i] == 1])
           action = int(input(f'{player}:'))

           if valid_moves[action] == 0:
               print("action not valid")
               continue

       else:
           neutral_state = game.change_perspective(state, player)
           valid_moves = game.get_valid_moves(neutral_state)
           print("valid_moves", [i for i in range(game.action_size) if valid_moves[i] == 1])
           action = int(input(f'{player}:'))
           if valid_moves[action] == 0:
               print("action not valid")
               continue

       state = game.get_next_state(state, action, player)

       value, is_terminal = game. get_value_and_terminated(state, action)
       if is_terminal:
            print(state)
            if value == 1:
                print(player, 'won')
            else:
                print('draw')
            break

       player = game.get_opponent(player)

def play_with_machine(game, player, args,device):
    model = ResNet(game, args['num_resBlocks'], args['num_hidden']).to(device)
    model.load_state_dict(torch.load('models/model_4_64_ConnectFour/model_7_ConnectFour.pth', map_location=device))

    mcts = MCTS(game, args, model)

    state = game.get_initial_state()

    model.eval()
    with torch.inference_mode():
        while True:
            print(state)
            if player == 1:
                valid_moves = game.get_valid_moves(state)
                print("valid_moves", [i for i in range(game.action_size) if valid_moves[i] == 1])
                action = int(input(f'{player}:'))

                if valid_moves[action] == 0:
                    print("action not valid")
                    continue
            else:
                neutral_state = game.change_perspective(state,player)
                mcts_probs = mcts.search(neutral_state)
                action = np.argmax(mcts_probs)
                print(f'{player}:{action}')

            state = game.get_next_state(state, action, player)

            value, is_terminal = game.get_value_and_terminated(state, action)
            if is_terminal:
                print(state)
                if value == 1:
                    print(player, 'won')
                else:
                    print('draw')
                break

            player = game.get_opponent(player)

def train_model(game, args):
    model = ResNet
    optimizer = torch.optim.Adam
    alphaZero = AlphaZero(model,optimizer,game,args)
    train_losses = alphaZero.learn()
    
    epochs = range(len(train_losses))

    plt.figure(figsize=(8, 11))

    plt.plot(epochs, train_losses, label='train_loss')
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

def plot_model_predictions(state,device):
    torch.manual_seed(42)
    state = game.get_next_state(state, 6, 1)
    state = game.get_next_state(state, 2, -1)
    state = game.get_next_state(state, 0, 1)
    state = game.get_next_state(state, 3, -1)
    print(state)

    encoded_state = game.get_encoded_state(state)
    print(encoded_state.shape)

    tensor_state = torch.tensor(encoded_state).unsqueeze(dim=0).to(device)
    print(tensor_state.shape)

    model = ResNet(game, args['num_resBlocks'], args['num_hidden']).to(device)
    model.load_state_dict(torch.load('models/model_4_64_ConnectFour/model_7_ConnectFour.pth',map_location=device))

    model.eval()
    with torch.inference_mode():
        policy, value = model(tensor_state)
        policy = policy.softmax(dim=1).squeeze(0).detach().cpu().numpy()
        value = value.cpu().item()
        print(policy, value)
        plt.bar(range(game.action_size), policy)
        plt.show()

if __name__ == '__main__':
    play_alone(game, player)
    #plot_model_predictions(state,device)
    #train_model(game, args)
    #play_with_machine(game,player,args,device)
