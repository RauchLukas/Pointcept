import argparse

import default_argument_parser

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config-file",
         required=True,
         help="path/to/config_file")
    parser.add_argument(
        "-g", "--num-gpus",
         required=True,
         help="number of GPUs")
    args = parser.parse_args()

    for k, v in vars(args).items():
        print(f"{k} : {v}")



if __name__ == "__main__":

    main()