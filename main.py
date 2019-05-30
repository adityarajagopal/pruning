import os
import random
import sys

# set python path so it points to src
appDir = os.path.split(os.getcwd())
userDir = os.path.split(appDir[0])
srcDir = os.path.split(userDir[0])
sys.path.append(srcDir[0])

import src.ar4414.pruning.app as applic

import argparse
import getpass

def parse_command_line_args() : 
    parser = argparse.ArgumentParser(description='PyTorch Pruning')
    parser.add_argument('--config-file', default='None', type=str, help='config file with training parameters')
    args = parser.parse_args()
    return args

def main() : 
    # parse config
    print('==> Parsing Config File')
    args = parse_command_line_args()
    
    username = getpass.getuser()

    if args.config_file != 'None' : 
        app = applic.Application(args.config_file)
    else : 
        raise ValueError('Need to specify config file with parameters')

    app.main()
        
main()
         
