import torch
import  numpy as np
import  scipy.stats
import  argparse
import os

from maml import MAML
from utils import load_data, extract_episode



def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h



def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    num_way, num_shot, num_query = args.n_way, args.k_spt, args.k_qry

    net_config = [
        ('linear', [64, 81]),
        ('relu', [True]),
        ('linear', [64, 64]),
        ('relu', [True]),
        ('linear', [64, 64]),
        ('relu', [True]),
        ('linear', [64, 64]),
        ('relu', [True]),
        ('linear', [1, 64]),
        ('relu', [True]),
    ]

    proto_config = [
        ('linear', [64, 81]),
        ('relu', [True]),
        ('linear', [64, 64]),
        ('relu', [True]),
        ('linear', [64, 64]),
        ('relu', [True]),
        ('linear', [64, 64]),
        ('relu', [True]),
    ]

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    #maml = MAML(args, net_config, proto_config).to(device)
    maml = MAML(args, net_config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    # batchsz here means total episode number
    data_train = load_data('../data/superconductors/not_cuprate.csv')
    data_test = load_data('../data/superconductors/cuprate.csv')

    min_loss = 100000

    for _ in range(args.epoch):

        loss_all_train = []
        for step in range(100):

            db = extract_episode(data_train, num_way, num_shot, num_query)
            loss = maml(db)
            loss_all_train.append(loss)
            if step % 100 == 99:
                loss = np.array(loss_all_train).mean(axis=0).astype(np.float16)
                print('step:', step, '\ttraining loss:', loss)
                loss_all_train = []

        loss_all_test = []
        for step in range(100):
            db = extract_episode(data_test, num_way, num_shot, num_query)
            loss = maml.finetunning(db)
            loss_all_test.append(loss)
        loss = np.array(loss_all_test).mean(axis=0).astype(np.float16)
        print('Test loss:', loss)
        if loss.min() < min_loss:
            min_loss = loss.min()
            print('This is the best model so far, saving.')
            results_dir = './results/'
            if not os.path.exists(results_dir):
                os.mkdir(results_dir)
            torch.save(maml.state_dict(), results_dir + 'best_model.pth')


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=100)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=10)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=10)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=1)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.002)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()

    main()
