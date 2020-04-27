# Project journal
## Implementing binary single output channel

|Start Date|End Date  |
|----------|----------|
|2020-04-25|2020-04-26|

### Description

Implemented binary prediction for the existing multi-Class attention-gated Unet
ie. the unet can now also output a single binary channel

### Delivrables

- [x] Dice loss with binary channel
- [x] Visualisation, prediction and plotting for binary channels

|2 output channels|1 output channel  |
|----------|----------|
|![2 Output Channels Dice Loss](./static/journal/dual_output_channel_loss.png "Dual output channel loss") | ![1 Output Channel Dice Loss](./static/journal/single_output_channel_loss.png "Single output channel loss")|

### Conclusion

- Unet is correctly implemented
- Multi-channel prediction remains superior in terms of convergence  

## Focal Tversky loss

|Start Date|End Date  |
|----------|----------|
|2020-04-26|2020-04-26|

Implemented focal Tversky (FT) loss function with multi-channel output.  

|FT-Loss over time|Dice over time with FT as loss  |
|----------|----------|
|![FT-Loss](./static/journal/focal_tversky_loss_over_time.svg "FT-Loss over time") | ![FT-Dice](./static/journal/focal_tversky_dice_over_time.svg "Dice over time with FT as loss")|

### Conclusion

- Focal tversky loss performs well
- Similar overall performance when comparing to dice loss with 2 output channels
- Maybe more accurate on small segments
- Maybe more prone to overfitting


# TODO

- Implement augmentation
- save loss plots
- implement combined loss
