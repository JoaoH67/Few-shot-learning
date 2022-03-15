## Things to do

1. Change pyplot to matplotlib
2. Right now: training on whole dataset of non-cuprates. Try training with only 50 examples (compare w/ classical methods, see 3).
3. Create a deep learning model to predict the temperature of the cuprates, using a training set of only 50 cuprates. Architecture: 
(look up in few_shot.py)
encoder = nn.Sequential(
        linear_block(x_dim (81, the features), hid_dim(64)),
        linear_block(hid_dim, hid_dim),
        linear_block(hid_dim, hid_dim),
        linear_block(hid_dim, hid_dim),
        linear_block(hid_dim, z_dim (1))
        )

## For the video


**Project Description**: Predicting Superconducting Critical Temperatures ($T_C$) using Few-Shot Learning Methods <br/>
**Course**: MENG 25620, Winter Quarter 2022 <br/>
**Group Members**: <br/>

