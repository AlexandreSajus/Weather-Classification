# Valeo Weather Classification
Classifying street-level images according to weather to help the training of autonomous vehicles

![](media/acdc_example.png)

## Strategy
- **Day/Night**: RandomForest on average RGB and HSV values
- **Clear/Rain/Snow**: Simple CNN with Dropout
- **NoFog/Fog**: Simple CNN

Training on a mix between two manually cleaned datasets:
- **ACDC** ( https://acdc.vision.ee.ethz.ch/ )
- **Berkeley DeepDrive** ( https://bdd-data.berkeley.edu/ )

## Performance
- **Day/Night**: 97.8% accuracy ( Human Level Performance = 100% )
- **Clear/Rain/Snow**: 92.0% accuracy ( HLP = 96% )
- **NoFog/Fog**: Not yet evaluated
  
Here is the Clear/Rain/Snow confusion matrix

<img src="media/cm_precipitation.jpg" alt="cm_precipitation" width="400"/>

## Setup
- clone the repository
- pip install -r requirements.txt

## Run
- Put images to test in the inference_images folder
- Run inference.py
- The results will be saved in inference_results.csv
