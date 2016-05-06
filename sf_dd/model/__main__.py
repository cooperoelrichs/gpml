import argparse
from . import make_data_set
from . import make_model


def main():
    args = parse_args()

    if args.make_data_set:
        if not args.project_directory:
            raise ValueError(
                "Project Directory (-pd) is required for making a data set")

        print('Making data set - project_dir: %s' % args.project_directory)
        make_data_set.run(args.project_directory)

    if args.make_model:
        if not args.project_directory:
            raise ValueError(
                "Project Directory (-pd) is required for making a model")

        print('\nMaking model - project_dir: %s' % args.project_directory)
        make_model.run(args.project_directory)

    print('Finished.')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-md', '--make_data_set',
        help='make data set', action='store_true'
    )

    parser.add_argument(
        '-mm', '--make_model',
        help='make model', action='store_true'
    )

    parser.add_argument(
        '-pd', '--project_directory',
        help='porject directory', type=str
    )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
