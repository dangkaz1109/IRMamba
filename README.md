


### I. Run Options
pip install -r requirements.txt
#### 1. Using Pretrained Models
*   **Main Pretrained Model:** [Download here](https://drive.google.com/file/d/1hpt-t7WOJ8QwU-1_zrRG2zgTIL7dLJFG/view?usp=sharing)
*  Then place to the 'data' folder
*   **DINO Pretrained Model:** [Download here](https://drive.google.com/file/d/1aBPftvO5Metttw0XaHrcriVA9UTsQ6qn/view?usp=sharing)
* Then place to the 'data/dino_pretrain' folder
#### 2. Training From Scratch
1.  **Train the DINO backbone:**
    ```bash
    python train_dino.py
    ```
2.  **Train the Decoder:**
    ```bash
    python train_decoder.py
    ```

---

## 📊 Evaluation & Testing
```bash
python evaluate.py
