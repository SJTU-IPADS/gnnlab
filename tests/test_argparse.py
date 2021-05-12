import argparse

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Test")
    argparser.add_argument('--pipeline', action='store_true', default=False)
    argparser.add_argument('--dataset-path', type=str,
                           default='/graph-learning/samgraph/papers100M')
    argparser.add_argument('--l', nargs='+', type=int, default=[1, 2, 3])

    args = argparser.parse_args()
    print(vars(args))
    li = args.l
    w = (f"{n}" for n in li)
    print(w)
    s = " ".join(w)
    print(len(s))
