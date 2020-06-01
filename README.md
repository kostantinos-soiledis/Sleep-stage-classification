# Sleep-stage-classification
In this challenge we perform sleep stage classification, by using EEG signals

# Deep Learning Architectures Used
Final Model:
Based on existing literature about training EEGs with CNNs, we built an architecture that combines three CNN models: a Conv1D for each channel of the raw signal, and another Conv2D that uses spectrogram coefficients extracted from the raw signals (see Appendix, Figure 1). The output vectors from pooling layers of the three models are concatenated, where we vote for the best result for each line. By doing this, we are averaging out the errors of each guess.

# Previous attempts and literature review:
We tried different approaches that resulted in our choice for a mixed model:
Many articles (1) (2) propose to build a CNN for the signals, as the data is organized in time series. We built Conv1D and Conv2D models using the two channels together and separately, but didn’t get satisfactory results. (~ 70% on validation)
Some sources (3) suggest using spectrograms for classification tasks because their time-frequency domain allows extracting information in the two domains simultaneously. We built a Conv1D model using spectrogram coefficients extracted from the raw signals, and we got better results (~ 79% on validation)
Others (4) (5) mention the use of the wavelet transformation as a more sophisticated method, but we decided to not implement it because of its complexity.

# Description of the Experiment
We downsampled the raw signals to be more computationally efficient. To do this, we tuned the parameters through gridsearch.
We separated the two channels of the raw signal and reshaped them to fit in each in Conv1D models
We performed a sort of feature engineering by creating our own spectrograms extracted from the raw data. 
We used the spectrogram_lspopt function to extract spectrograms more resolute than those given
We used power_to_db, which boosts the spectrogram coefficients and make them more distinguishable
We built three independent CNN models, two Conv1D using each channel of the raw signals, and one Conv2D using spectrograms
For all models, we used the activation functions ReLU for the hidden layers and Softmax for the output layers, as they are the standard for multiclass classification 
The number of layers and neurons within the layers was optimized with random searching
We decided to use Adam optimizer. Even though AMSgrad yielded faster conversion, Adam provided better results in per-class and overall accuracy.
Within Keras API framework, we combined the two models by concatenating the output vectors from the Maxpooling layer of each model. By concatenating, the full range of features from both raw signals and spectrograms are made available to the classification layers, thus improving accuracy and generalisability. 

Results: see Appendix, Figure 2 & Table 1

# Discussion of the Solution’s Performance
Our main challenge was having only two channels of information, while most cases in literature benefit from having more channels available. Also, some classes have a frequency overlapping by definition, so we tried to distinguish them as much as possible through feature engineering. Therefore, we chose to mix CNN models of raw signals and spectrograms, looking to tackle the overlap in classes and the noise in the signals. One strong advantage of the CNN model is the kernels because  they enable temporal features within EEG’s to be learnt.
We admit there is room for improvement. One approach that could yield better results would be by further hyperparameter tuning of layers and kernel sizes. Also, another architecture option could be to combine the CNN with LSTM, which processes features extracted by the latter (2). However, we didn’t take that path since our training time was already taking too long, and we wanted to experiment with our mixed models approach. 

Score: 0.463 - not as expected regarding our validation score (0.80), actually lower than a previous submission of a Conv1D using spectograms only (0.639)
Appendix

Figure 1. Representation of the mixed CNN models of spectograms (left), one channel of raw signal (center) and the other (right)




Figure 2. Confusion Matrix of normalized classes






Table 1. Classification metrics per class



Validation Accuracy: 0.80
Loss: 0.52



# References
Cui, Zhihong et al. “Automatic Sleep Stage Classification Based on Convolutional Neural Network and Fine-Grained Segments.” Hindawi, 8 Oct. 2018, hindawi.com/journals/complexity/2018/9248410/.
Mansar, Youness. “Sleep Stage Classification from Single Channel EEG using Convolutional Neural Networks.” Towards Data Science, 1 Oct. 2018, towardsdatascience.com/sleep-stage-classification-from-single-channel-eeg-using-convolutional-neural-networks-5c710d92d38e.
Ramos, Ricardo et al. “Analysis of EEG Signal Processing Techniques based on Spectrograms.” Research in Computing Science 145, 2017, rcs.cic.ipn.mx/2017_145/Analysis%20of%20EEG%20Signal%20Processing%20Techniques%20based%20on%20Spectrograms.pdf.
Taspinar, Ahmet. “A guide for using the Wavelet Transform in Machine Learning.” Ahmet Taspinar, 21 Dec. 2018, ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/.
Ubeyli, Elif. “Combined neural network model employing wavelet coefficients for EEG signals classification.” Digital Signal Processing, Mar. 2009. sciencedirect.com/science/article/abs/pii/S105120040800122X.

