# Speech Separation
The simple project to separate mixed voice (2 clean voices) to 2 separate voices.


**Result Example (Clisk to hear the voices)**: 
[mix](output_sample/mix.mp3) ||  [prediction voice1](output_sample/pred1.mp3) || [prediction voice2](output_sample/pred2.mp3)

*Mix Spectrogram*

![mix](https://user-images.githubusercontent.com/19920599/132132582-0c504d5a-935c-484b-90db-300735cf206b.png)

*Predict Voice1's Spectrogram*

![pred1](https://user-images.githubusercontent.com/19920599/132132632-830b5826-230d-4c0f-96c3-c4b8f9d3b146.png)

*Predict Voice2's Spectrogram*

![pred2](https://user-images.githubusercontent.com/19920599/132132678-d7f2a12b-8e9b-416d-b057-03bb915be38b.png)


## 1. Quick train
### Step 1:
Download [LibriMixSmall](https://zenodo.org/record/3871592/files/MiniLibriMix.zip?download=1 ), extract it and move it to the root of the project. 
### Step 2:
`./train.sh`

It will take about **ONLY 2-3 HOURS** to train with normal GPU. After each epoch, the prediction is generated to `./viz_outout` folder.

## 2. Quick inference
`./inference.sh`
The result will be generated to `./viz_outout` folder.



## 3. More detail
+ **Input**:
   The Complex spectrogram. Get from the raw mixed audio signal
+ **Output**:
   The complex ratio mask (cRM) ---> complex spectrogram ---> separated voices.
+ **Model**:
  Use the simple version of [this implementation](https://github.com/bill9800/speech_separation/blob/master/model/lib/model_AO.py) , which is defined in paper [Looking to Listen at the Cocktail Party: A Speaker-Independent Audio-Visual Model for Speech Separation](https://arxiv.org/abs/1804.03619)

+ **Loss function**:
  [Permutation Invariant Training Loss](https://arxiv.org/pdf/1607.00325.pdf) and [PairWise Neg SisDr Loss](https://openreview.net/pdf?id=SkeRTsAcYm) (more SOTA)
  
+ **Dataset**:
 A small version of `LibriMix` dataset. I get from [LibriMixSmall](https://zenodo.org/record/3871592/files/MiniLibriMix.zip?download=1) 
 
 ## 4. Current problem
 Due to small dataset size for fast training, the model is a bit overfitting to the training set. Use the bigger dataset will potentially help to overcome that.
 Some suggestions: 
 1. Use the [original LibriMix Dataset](https://github.com/JorisCos/LibriMix) which is way much bigger (around 60 times bigger that what I have trained).
 2. Use [this work](https://github.com/bill9800/speech_separation/tree/master/data/audio) to download much more in-the-wild dataset and use `datasets/VoiceMixtureDataset.py` instead of the Libri one that I am using. p/s I have trained and it work too.


