# Denotia
Denotia is a web app to predict rare neurological diseases from MRI scans.

[Web Application](https://denotia.herokuapp.com/) | [MIT Solve Submission](https://solve.mit.edu/challenges/horizonprize/solutions/51857)

Software used: Python, Freesurfer, GrayNet, PyTorch Geometric, Bootstrap, Heroku.

## Inspiration
Diagnosis of rare diseases is majorly dependent on professional knowledge and clinical experience. Tripled with low doctor-patient ratios and overburdened diagnostic clinics, this often leads to an unsatisfactory diagnostic accuracies. At the same time, recent AI-driven interventions to discover high-quality imaging biomarkers have been extremely successful.

## Application
Denotia can classify MRI scans into subtypes of frontotemporal dementia (FTD) and Alzheimer's disease (AD) within a few seconds. Our model uses a novel deep-learning method that analyzes these scans based on the differences between different structural regions of the cortex, with 86% and 84% accuracy respectively. This helps reduce premature mortality and improve quality of care.

## Program Design
Denotia employs an image-classifying deep learning model built on Python and trained using scans from FTLDNI, the largest database for scans of frontotemporal lobar degeneration (FTLD). The model is capable of recognizing surrogate markers of FTLD and continuously improve.

1.Cortical thicknesses are automatically extracted from uploaded scans using a preprocessing suite. This saves time and resources and prevents most manual errors.
The thickness data is transformed into a network graph to serve as the data point for the algorithm.
2.A trained, strongly optimized graph neural network model is used to classify uploaded network graphs based on regional edge-weight variations in the graph.
3.After several rounds of feature extractions are completed in a few milliseconds, the final result is displayed as FTD positive/negative and AD positive/negative, along with the predicted sub-type.
4. Early classification leads to efficient streamlining of interventions and treatment plans. Long-term symptom management and treatment approaches can be finalized thanks to an accurate and expedited diagnosis. Moreover, this can accelerate pharmacological interventions and clinical studies through effective patient selection, stratification, and real-time measurement of outcomes.
## Diagnostic Accuracy
Is the diagnosis confirmed? Partly. For neurological diseases, both clinical and radiological validation is required. Our tool helps radiologists provide quantitative evidence to corroborate clinical findings. Contact your healthcare practitioner for detailed guidance.

## Software
We employed the following software while building Denotia.

- FreeSurfer and GrayNet to process T1-weighted MRI scans from FTLDNI. This helps extract the cortical thickness, and then converted into 2D network graphs.
- PyTorch Geometric to build the GNN. This comprises of two sets: the GraphSAGE layers, and the dense differentiable pooling layers.
- Bootstrap and Heroku to deploy the web app for end-user access.
