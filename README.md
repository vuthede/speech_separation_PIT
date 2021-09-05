# Speech Separation
The simple project to separate mixed voice (2 clean voice) to 2 separate voices.

+ Input:
   The Complex spectrogram. Get from the raw signal
+ Output:
   The complex ratio mask (cRM). Then We can infer the complex spectrogram of each single voice. Then we can get the separated voices.
+ Model:
  Use the simple version of [this implementation](https://github.com/bill9800/speech_separation/blob/master/model/lib/model_AO.py) , which is defined in paper [Looking to Listen at the Cocktail Party: A Speaker-Independent Audio-Visual Model for Speech Separation](https://arxiv.org/abs/1804.03619)

+ Loss function:
  [Permutation Invariant Training Loss](https://arxiv.org/pdf/1607.00325.pdf)

+ Dataset
 A small version of `LibriMix` dataset. I get from [LibriMixSmall](https://zenodo.org/record/3871592/files/MiniLibriMix.zip?download=1) 

## Quick train and test
### Step 1:
Download [LibriMixSmall](https://zenodo.org/record/3871592/files/MiniLibriMix.zip?download=1 ), extract it and move it to the root of the project. 
### Step 2:
`./train.sh`
It will take about **2-3 HOURS** to train with normal GPU

