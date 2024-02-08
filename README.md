# Adjusting-parameters-of-PID-with-Neural-Network.
This project uses BPNN to dynamically adjust the parameters of the PID algorithm.

Use Pytorch to build a simple network architecture, and then the loss function is customized. The overall idea is to use the current measurement value, target value, and error as the three-dimensional input of the network, and then the PID parameters KP, KI, and KD are the three-dimensional output of the network. The output of this network is passed to the incremental PID algorithm. The loss of the entire network can be defined as the output of the incremental PID. The goal is to use the backpropagation algorithm of gradient descent to reduce the loss of the network (incremental PID output) to 0, which is the actual arrival of the control system. steady state.
​**However, the network is unstable, possibly due to parameter tuning problems of the neural network. You can run it several times to get a better result. In the ``result`` folder, the better results of my operation are shown.**
### The role of each file
``PID_INC.py`` This is an algorithm of PID.

``pid_control.ipynb`` This is a demo of using PID algorithm.

``bp_pid.ipynb`` This is a demo of using BP_PID algorithm.

``compare.py`` This is a demo of plotting the result of PID and BP_PID.
### Configure the following python environment
pandas、numpy、torch、matplotlib
