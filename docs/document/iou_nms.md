#### 🌟IoU loss function
使用
* **CIoU（default）**
```python
python train.py --loss_category CIoU
```
* **DIoU**
```python
python train.py --loss_category DIoU
```
* **GIoU**
```python
python train.py --loss_category GIoU
```
* **EIoU**
```python
python train.py --loss_category EIoU
```
* **SIoU**
```python
python train.py --loss_category SIoU
```
______________________________________________________________________

#### 🌟NMS
使用
* **NMS（default）**
```python
python val.py
```
* **Merge-NMS**
```python
python val.py --merge
```
* **Soft-NMS**
```python
python val.py --soft
```
______________________________________________________________________

