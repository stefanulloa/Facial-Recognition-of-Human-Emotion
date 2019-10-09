# Facial Recognition of Human Emotion
 CS Eng. Bacherlor's Thesis on Deep learning, Affective Computing and Computer Vision

The purpose of this thesis work is to try to surpass current visual emotion detection systems, due to the huge potential affective computing (recognition or simulation of human emotion) could have. If computing systems could adequate their behaviour according to the user's emotions then the interaction between both would be much more succesful/beneficial. Before that we need computing systems to detect emotions sufficiently accurately. In order to do so, we take a look at the state-of-the-art and develop neural networks systems so that they can get better results.

### We work with two datasets: 
*   AFEW-VA https://ibug.doc.ic.ac.uk/resources/afew-va-database/
*   SEMAINE https://semaine-db.eu/

Python 3.
PyTorch was used for the neural networks code.

The models where trained (and validated) on UPF' high computing processing (HCP) system.

The models where subjectively tested with a few samples.

For testing, a face detection software (not included in the repo; also used to generate SEMAINE dataset only-face images) by D. Aspandi from UPF was used.   

### In this repository you will find:
*   The data processing files. This access the datasets and generates files for images, V-A values and facial landmark points.
*   The neural networks training and validating code.
*   The instructions sh file for the HCP to run the training code.

Results and findings on the thesis document.