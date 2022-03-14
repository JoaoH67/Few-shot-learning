import torch
import  numpy as np
import  scipy.stats
import  argparse

from meta import Meta
from data_loader import load_data, extract_episode


def obtain_data(data, num_way, num_shot, num_query, device):
    db = extract_episode(data, num_way, num_shot, num_query)
    features, temps = db['features'].to(device), db['temps'].to(device)
    # from each class, extract num_shot support images
    x_support = features[:, :num_shot]  # lines are classes and columns are images
    # from each class, extract the remaining images as query images
    x_query = features[:, num_shot:]  # lines are classes and columns are images
    # transform into a array in which all images are contiguous
    x_support = x_support.contiguous().view(1, num_way * num_shot, -1)  # no more lines and columns
    x_query = x_query.contiguous().view(1, num_way * num_query, -1)

    y_support = temps[:, :num_shot].to(device)
    y_query = temps[:, num_shot:].to(device)
    y_support = y_support.contiguous().view(1, num_way * num_shot, 1)
    y_query = y_query.contiguous().view(1, num_way * num_query, 1)

    return x_support, y_support, x_query, y_query



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

    config = [
        ('linear', [32, 81]),
        ('relu', [True]),
        ('linear', [16, 32]),
        ('relu', [True]),
        ('linear', [8, 16]),
        ('relu', [True]),
        ('linear', [1, 8]),
        ('relu', [True])
    ]

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    # batchsz here means total episode number
    data_train = load_data('../data/superconductors/not_cuprate.csv')
    data_test = load_data('../data/superconductors/cuprate.csv')

    for _ in range(args.epoch):

        loss_all_train = []
        for step in range(900):

            x_support, y_support, x_query, y_query = obtain_data(data_train, num_way, num_shot, num_query, device)
            loss = maml(x_support, y_support, x_query, y_query)
            loss_all_train.append(loss)
            if step % 300 == 299:
                loss = np.array(loss_all_train).mean(axis=0).astype(np.float16)
                print('step:', step, '\ttraining loss:', loss)
                loss_all_train = []

        loss_all_test = []
        for step in range(300):
            x_support, y_support, x_query, y_query = obtain_data(data_test, num_way, num_shot, num_query, device)
            loss = maml.finetunning(x_support, y_support, x_query, y_query)
            loss_all_test.append(loss)
        loss = np.array(loss_all_test).mean(axis=0).astype(np.float16)
        print('Test los:', loss)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=100)
    argparser.add_argument('--n_way', type=int, help='n way', default=2)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=10)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=10)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=1)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()

    main()
