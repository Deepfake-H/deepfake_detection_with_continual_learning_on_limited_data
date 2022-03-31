import shlex
import subprocess
import time

commands = []

# commands.append({'name': 'Run 1-4',
#                  'cmd': 'python ./run.py --training_set GAN-S VFHQ ForenSynths DeepFake '
#                         '--test_set GAN-S VFHQ ForenSynths DeepFake NeuralTextures '
#                         '--continual_learning NoCL'})
#
# commands.append({'name': 'Run 5,6',
#                  'cmd': 'python ./run.py --training_set GAN-S VFHQ ForenSynths DeepFake '
#                         '--test_set GAN-S VFHQ ForenSynths DeepFake NeuralTextures '
#                         '--continual_learning Normal'})

commands.append({'name': 'Run iCarl',
                 'cmd': 'python ./run.py --training_set GAN-S VFHQ ForenSynths DeepFake '
                        '--test_set GAN-S VFHQ ForenSynths DeepFake NeuralTextures '
                        '--continual_learning CL --cl_type ni --agent ICARL '
                        '--retrieve random --update random --mem_size 5000'})

commands.append({'name': 'Run GDumb',
                 'cmd': 'python ./run.py --training_set GAN-S VFHQ ForenSynths DeepFake '
                        '--test_set GAN-S VFHQ ForenSynths DeepFake NeuralTextures '
                         '--continual_learning CL --cl_type=ni --agent GDUMB '
                         '--mem_size 3000 --mem_epoch 10 --minlr 0.0005 --clip 10'})

commands.append({'name': 'Run LWF',
                 'cmd': 'python ./run.py --training_set GAN-S VFHQ ForenSynths DeepFake '
                        '--test_set GAN-S VFHQ ForenSynths DeepFake NeuralTextures '
                        '--continual_learning CL --cl_type=ni --agent LWF '})

commands.append({'name': 'Run MIR',
                 'cmd': 'python ./run.py --training_set GAN-S VFHQ ForenSynths DeepFake '
                        '--test_set GAN-S VFHQ ForenSynths DeepFake NeuralTextures '
                        '--continual_learning CL --cl_type=ni --agent ER '
                        '--retrieve MIR --update random --mem_size 5000'})

for i, comm in enumerate(commands):
    run_start = time.time()
    print('\n@@@ RUN ALL @@@   -- job {} -- {} -- Start\n'.format(i, comm['name']))
    print(comm['cmd'])

    p = subprocess.Popen(shlex.split(comm['cmd']))
    p.wait()

    run_end = time.time()
    print("\n@@@ RUN ALL @@@   -- job {} -- {} -- END (run time {}s)\n".format(i, comm['name'], run_end - run_start))
    time.sleep(10)
