from tensorboard import program
import webbrowser
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-d','--logdir', help=' logdir',type=str,default='./')
args = parser.parse_args()

logdir = args.logdir
# learning visualizer
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', logdir])
url = tb.launch()
print("[RAISIM_GYM] Tensorboard session created for <" + logdir + " at: " + url)

while True:
    pass