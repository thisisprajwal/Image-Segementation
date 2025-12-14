#Image Segmentation
In this lab you will work through a series of exercises performing image segmentation, also called semantic segmentation. Semantic segmentation is the task of placing each pixel into a specific class. In a sense it's a classification problem where you'll classify on a pixel basis rather than an entire image. In this lab the task will be classifying each pixel in a cardiac MRI image based on whether the pixel is a part of the left ventricle (LV) or not.

This lab is not an introduction to deep learning, nor is it intended to be a rigorous mathematical formalism of convolutional neural networks. We'll assume that you have at least a passing understanding of neural networks including concepts like forward and backpropagation, activations, SGD, convolutions, pooling, bias, and the like. It is helpful if you've encountered convolutional neural networks (CNN) already and you understand image recognition tasks. The lab will use Google's TensorFlow machine learning framework so if you have Python and TensorFlow experience it is helpful, but not required. Most of the work we'll do in this lab is not coding per se, but setting up and running training and evaluation tasks using TensorFlow.

#Input Data Set
The data set you'll be utilizing is a series of cardiac images (specifically MRI short-axis (SAX) scans) that have been expertly labeled.

Introduction to Image Segmentation with TensorFlow
There are a variety of important image analysis deep learning applications that need to go beyond detecting individual objects within an image and instead segment the image into spatial regions of interest. For example, in medical imagery analysis it is often important to separate the pixels corresponding to different types of tissue, blood or abnormal cells so that we can isolate a particular organ. In this self-paced, hands-on lab we will use the TensorFlow machine learning framework to train and evaluate an image segmentation network using a medical imagery dataset.

#Objectives
By the time you complete this notebook you will be able to:

Understand how Neural Networks can solve imaging problems
Use Transpose Convolutional Neural Networks
Use Keras and TensorFlow 2 to analyze image data

#Deep Learning with TensorFlow
This lab is part of a series of self-paced labs designed to introduce you to some of the publicly-available deep learning frameworks available today. TensorFlow is a framework developed by Google and used by numerous researchers and product groups within Google.

TensorFlow is an open source software library for machine intelligence. The computations are expressed as data flow graphs which operate on tensors (hence the name). If you can express your computation in this manner you can run your algorithm in the TensorFlow framework.

TensorFlow is portable in the sense that you can run on CPUs and GPUs and utilize workstations, servers, and even deploy models on mobile platforms. At present TensorFlow offers the options of expressing your computation in either Python or C++, with varying support for other languages as well. A typical usage of TensorFlow would be performing training and testing in Python and once you have finalized your model you might deploy with C++.

TensorFlow is designed and built for performance on both CPUs and GPUs. Within a single TensorFlow execution you have lots of flexibility in that you can assign different tasks to CPUs and GPUs explicitly if necessary. When running on GPUs TensorFlow utilizes a number of GPU libraries including cuDNN allowing it to extract the most performance possible from the very newest GPUs available.

One of the intents of this lab is to gain an introductory level of familiarity with TensorFlow. In the course of this short lab we won't be able to discuss all the features and options of TensorFlow but we hope that after completion of this lab you'll feel comfortable with and have a good idea how to move forward using TensorFlow to solve your specific machine learning problems.

For comprehensive documentation on TensorFlow we recommend the TensorFlow website.


#TensorFlow Basics
TensorFlow 2 has introduced a number of significant updates from TensorFlow 1.X. A discussion of all of these changes is outside the scope of this lab, but one major change in particular is the that the default behavior of TensorFlow 2 is to execute in so-called eager mode. In TensorFlow 1.X, one has to construct a model, which is essentially a dataflow graph. Once this is complete you can then launch a Session to run the training data through the model. TensorFlow 2.0 updates this paradigm to execute in eager mode, which means that commands are executed in the order they are called, as a typical Python program would be expected to do.

This lab will illustrate the use of the tf.keras library, which is TensorFlow's implementation of the Keras API specification. Keras is a high-level neural network API, which, according to the Keras website, "was developed with a focus on enabling fast experimentation". It is easy to create and train a network in Keras as it has many common neural network layer types built in, e.g., fully-connected layers, convolutions, drop-out, pooling and RNN's.

Keras provides an easy mechanism to connect layers together. There is a Sequential model which, as the name suggests, allows one to build a neural network from a series of layers, one after the other. If your neural network structure is more complicated, you can utilize the Functional API, which allows more customization due to non-linear topology, shared layers, and layers with multiple inputs. And you can even extend the Functional API to create custom layers of your own. These different layer types can be mixed and matched to build whatever kind of network architecture you like. Keras provides a lot of great built-in layers types for typical use cases, and allows the user to build complex layers and models as your needs dictate. These layers are backed by TensorFlow under-the-covers so the user can concern themself with their model and let TensorFlow worry about performance.

#Sample Workflow
A sample workflow of training and evaluating a model might look like the following.

Prepare input data--Input data can be Numpy arrays but for very large datasets TensorFlow provides a specialized format called TFRecords.
Build the Keras Model--Structure the architecture of your model by defining the neurons, loss function, and learning rate.
Train the model--Inject input data into the TensorFlow graph by using model.fit. Customize your batch size, number of epochs, learning rate, etc.
Evaluate the model--run inference (using the same model from training) on previously unseen data and evaluate the accuracy of your model based on a suitable metric.
TensorBoard
TensorFlow provides a feature-rich tool called TensorBoard that allows you to visualize many aspects of your program. In TensorBoard, you can see a visual representation of your computation graph and you can plot different metrics of your computation such as loss, accuracy, and learning rate. Essentially any data that is generated during the execution of TensorFlow can be visually displayed by TensorBoard with the addition of a few extra API calls in your program.

Tensorboard hooks into Keras through a training callback

#TensorBoard
TensorFlow provides a feature-rich tool called TensorBoard that allows you to visualize many aspects of your program. In TensorBoard, you can see a visual representation of your computation graph and you can plot different metrics of your computation such as loss, accuracy, and learning rate. Essentially any data that is generated during the execution of TensorFlow can be visually displayed by TensorBoard with the addition of a few extra API calls in your program.

Tensorboard hooks into Keras through a training callback
