# WOMANIUM_PROJECT_2024_DRAGON-S_HEART
 ->TEAM NAME: DRAGON'S HEART

->TEAM MEMBERS: 
1. SAMEER PAREEK (WOMANIUM ID: WQ24-8fN4YFezDqkq0sY)
2. VAIBHAVI YADAV (WOMANIUM ID: WQ24-q1TzvnPoNi9KKkV)

----

Project Task 1: Familiarizing with Pennylane

In this task, we delved into Pennylane, a powerful tool for quantum computing, by exploring key sections of the Pennylane Codebook. Here's a breakdown of our journey:
1. Introduction to Quantum Computing:
•	We started with the fundamentals of quantum computing, where we learned about qubits, the basic unit of quantum information, and how they differ from classical bits.
•	The section likely covered the concept of superposition, where a qubit can exist in multiple states simultaneously, and entanglement, where qubits become interconnected in ways that defy classical physics.
2. Single-Qubit Gates:
•	Moving on to operations on qubits, we explored single-qubit gates, which are the quantum equivalent of logical operations in classical computing.
•	We learned about different types of single-qubit gates such as the Pauli-X, Y, and Z gates, the Hadamard gate (H), and the phase gate (S), and how these gates manipulate the state of a qubit.
3. Circuits with Many Qubits:
•	Finally, we tackled circuits involving multiple qubits, where the real power of quantum computing becomes evident.
•	We learned how to construct and simulate quantum circuits that involve entangling multiple qubits, using multi-qubit gates like the CNOT (Controlled-NOT) gate and understanding how these circuits can be used to perform complex computations that are challenging for classical computers.
Throughout these sections, we gained hands-on experience with Pennylane by coding and simulating quantum circuits. This practical approach solidified our understanding of quantum mechanics' foundational concepts and their application in quantum computing.
This task has given us a solid foundation in quantum computing principles and the tools needed to create and manipulate quantum circuits using Pennylane.

Document of our progress and learnings from these Pennylane tutorials are attached as DRAGON’S_HEART_TASK_1.



Project Task 2: Exploring Quantum Machine Learning with a Variational Classifier

In this task, we dove into the fascinating world of Quantum Machine Learning (QML) by working through a tutorial on building a Variational Classifier. Here's a breakdown of the process and the purpose of each step:
1. Understanding the Variational Classifier:
•	We started by grasping the concept of a Variational Classifier, which is a type of quantum machine learning model. Unlike traditional classifiers, this one leverages the power of quantum circuits to make predictions based on data.
•	The classifier is built using parameterized quantum circuits, where the parameters are adjusted during training to minimize the classification error.
2. Data Preprocessing:
•	Like any machine learning task, we began by preparing our data. In the quantum context, this involved encoding classical data into a quantum state—a crucial step as it bridges the classical and quantum realms.
•	We used 2 examples for explaining our variational quantum classifier so for first example we had two datasets i.e. train and test(TASK_2_Example1_TRAIN and TASK_2_Example1_TEST) and for second example we had one IRIS dataset i.e. TASK_2_Example2_IRIS. Create an empty folder in our system and then put thse dataset in that folder, copy the path of this folder specified in code
•	This step ensures that our data is compatible with quantum operations, laying the groundwork for the quantum model to process it.
3. Building the Quantum Circuit:
•	We then constructed a quantum circuit that would act as the heart of our classifier. This involved setting up quantum gates that could be tuned (via parameters) to perform the classification task.
•	The circuit was designed to evolve the quantum state based on the input data, with the goal of distinguishing between different classes.
4. Defining the Cost Function:
•	To train the classifier, we defined a cost function—a measure of how well the classifier is performing. In this context, the cost function typically represents the difference between the predicted and actual labels of our data.
•	The goal was to minimize this cost function by adjusting the parameters of the quantum circuit, thereby improving the accuracy of the classifier.
5. Training the Classifier:
•	The training process involved running an optimization algorithm to find the best parameters for the quantum circuit. This step is analogous to training in classical machine learning, where we iteratively improve the model by feeding it data and updating its parameters.
•	The optimizer used the gradients of the cost function to guide the parameter adjustments, ultimately leading to a well-trained quantum classifier.
6. Evaluating the Model:
•	After training, we evaluated the performance of our quantum classifier on a test dataset. This step was crucial to determine whether the model had successfully learned to classify the data or if it needed further tuning.
•	Evaluation metrics such as accuracy or loss were used to quantify the model’s performance.
By working through these steps, we’ve gained a solid understanding of the basic workflow in Quantum Machine Learning. We’ve not only implemented a variational classifier but also learned how each step contributes to the overall goal of building a quantum model capable of making accurate predictions.

Implementation and presentation  of the usual steps in this workflow with the  purpose of each step are explained in your own words are attached as DRAGON’S_HEART_TASK_2.



Project Task 3: Exploring Quanvolutional Neural Networks with the MNIST Dataset

In this task, we took on the challenge of implementing a more sophisticated quantum model by exploring Quanvolutional Neural Networks (QNNs) using the MNIST dataset, which is a standard dataset of handwritten digits used in machine learning.
Here's a breakdown of what we did and the key insights from each step:
1. Understanding the Quanvolutional Neural Network (QNN):
•	We started by understanding the concept of QNNs, which combine classical convolutional neural networks (CNNs) with quantum computing principles.
•	The idea was to use quantum circuits to enhance feature extraction, where the quantum operations can potentially capture complex patterns in data that classical methods might miss.
2. Data Preparation - MNIST Dataset:
•	We used the MNIST dataset(named as mnist.npz), a collection of 28x28 pixel images of handwritten digits. The dataset is commonly used for image classification tasks.
•	Our first step was to preprocess the data, normalizing and formatting it to be fed into the quantum circuits. This was crucial for ensuring that the data was in a state that could be efficiently processed by both classical and quantum components of our model.
3. Building the Quanvolutional Layer:
•	The core of the QNN was the Quanvolutional layer, where we used quantum circuits to process the image data.
•	This layer acted as a feature extractor, where small patches of the input image were passed through a quantum circuit. The output of this circuit was then used as features for the next stage of the network.
•	The quantum circuits in this layer were designed to capture intricate patterns in the image data, potentially offering an advantage over purely classical methods.
4. Integrating with a Classical Neural Network:
•	After the quantum feature extraction, we integrated these quantum-derived features with a classical neural network.
•	The classical network took these features and further processed them to classify the images into their respective digit categories. This hybrid approach leveraged the strengths of both quantum and classical computation.
5. Training the Model:
•	We trained the entire Quanvolutional Neural Network on the MNIST dataset, adjusting the parameters of both the quantum circuits and the classical layers.
•	The training process was iterative, with the goal of minimizing the classification error on the dataset. This involved fine-tuning the quantum circuits to extract the most informative features from the image data.
6. Evaluating the Model:
•	Finally, we evaluated the performance of the QNN on a test set from the MNIST dataset. This step was critical in determining how well our quantum-enhanced model performed compared to classical benchmarks.
•	We analyzed metrics such as accuracy and loss to understand the strengths and potential areas for improvement in our QNN.
By working through these steps, we gained valuable experience in implementing a hybrid quantum-classical model and explored how quantum computing could enhance traditional machine learning tasks.
Implementation and presentation of our steps in a notebook and commenting on the important steps are attached as DRAGON'S_HEART_TASK_3.
