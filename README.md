# SS-VBT


## Installation
The model is built in Python3.8.5, PyTorch 1.12.1 in Ubuntu 18.04 environment.

## Data Preparation

### 1. Prepare Training Dataset

Please put your dataset under the path: **./SS-VBT/data/train**.


### 2. Prepare Validation Dataset

​	Please put your dataset under the path: **./SS-VBT/data/validation**.

## Pretrained Models

The pre-trained models are placed in the folder: **./SS-VBT/pretrained_models**

```yaml
# # For synthetic denoising
# gauss25
./pretrained_models/g25.pth

# # For field data denoising
./pretrained_models/Field.pth



## Train
* Train on synthetic dataset
```shell
python train_ssvbt.py --noisetype gauss25 --data_dir ./data/train/Imagenet_val --val_dirs ./data/validation --save_model_path ../experiments/results --log_name b2u_unet_gauss25_112rf20 --Lambda1 1.0 --Lambda2 2.0 --increase_ratio 20.0
```
## Test

* Test on **SEG_C3,BP2007**

  * For noisetype: gauss15

    ```shell
    python test_ssvbt.py --noisetype gauss15 --checkpoint ./pretrained_models/g15.pth --test_dirs ./data/validation --save_test_path ./test --log_name b2u_unet_g15_112rf20 --beta 19.7
    ```
 * For noisetype: gauss25

    ```shell
    python test_ssvbt.py --noisetype gauss25 --checkpoint ./pretrained_models/g25.pth --test_dirs ./data/validation --save_test_path ./test --log_name b2u_unet_g25_112rf20 --beta 19.7
    ```
* For noisetype: gauss50

    ```shell
    python test_ssvbt.py --noisetype gauss50 --checkpoint ./pretrained_models/g50.pth --test_dirs ./data/validation --save_test_path ./test --log_name b2u_unet_g50_112rf20 --beta 19.7
    ```

* Test on **Mobil Viking Graben Line 12 dataset**

  *  For Confocal_FISH

    ```shell
    python test_ssvbt.py --checkpoint ./pretrained_models/Field.pth --test_dirs ./dataset/FieldData --subfold Confocal_FISH --save_test_path ./test --log_name Confocal_FISH_b2u_unet_FIELD_112rf20 --beta 20.0
    ```
### Network architecture
![RUNOOB 图标](./networkandresult/1.png)

### Test Results
#### 1. ADNet for BSD68
![RUNOOB 图标](./networkandresult/2BSD.png)

#### 2. ADNet for Set12
![RUNOOB 图标](./networkandresult/3Set12.png)

#### 3. ADNet for CBSD68, Kodak24 and McMaster
![RUNOOB 图标](./networkandresult/4color.png)

#### 4. ADNet for CBSD68, Kodak24 and McMaster
![RUNOOB 图标](./networkandresult/5realnoisy.png)

#### 5. Running time of ADNet for a noisy image of different sizes.
![RUNOOB 图标](./networkandresult/6ruungtime.png)

#### 6. Complexity of ADNet
![RUNOOB 图标](./networkandresult/7complexity.png)

#### 7. 9 real noisy images
![RUNOOB 图标](./networkandresult/8realnoisy.png)

#### 8. 9 thermodynamic images from the proposed A
![RUNOOB 图标](./networkandresult/9ab.png)

#### 9. Visual results of BSD68
![RUNOOB 图标](./networkandresult/9gray.png)

#### 10. Visual results of Set12
![RUNOOB 图标](./networkandresult/10gray.png)

#### 11. Visual results of Kodak24
![RUNOOB 图标](./networkandresult/11.png)

#### 12. Visual results of McMaster 
![RUNOOB 图标](./networkandresult/12.png)

 
