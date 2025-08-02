# How to Run the AI-Trained Game Screenshot Upscaler Project

📄 ** Note: [View Full Project Report (PDF)](./📝PROJECT_REPORT.pdf)**

This project provides three simple ways to explore the super-resolution capabilities of the trained EDSR model on game screenshots. You can:

1. **Run the Gradio demo locally**
2. **Test specific image outputs using `test_sr_model.py`**
3. **View results in the final report with before-and-after comparisons**

---

## 1. Run the Demo UI

This is the most user-friendly way to try the model. A simple interface will let you upload a low-res game screenshot and see the enhanced output.

### Steps:
1. **Install dependencies** (once):
   pip install -r requirements.txt
2. **Run the Demo script**:
   python demo.py
3. **A local URL will appear (e.g., http://127.0.0.1:7860). Open it in your browser.**

4. **Upload a game screenshot and view the super-resolved result next to the original.**

## 2. Test a Specific Image with test_sr_model.py

This method lets you test and visualize how a single image passes through the model.

### Steps:
1. **Open test_sr_model.py**:

2. **Go to line 21 and change the index to any number to test a different image then run the script**:
   python test_sr_model.py
3. **python test_sr_model.py**

## 3. View the Final Report

The final report includes visual results, training notes, and architecture details. Before and after images are provided.




