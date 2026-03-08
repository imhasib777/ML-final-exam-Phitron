import gradio as gr
import pandas as pd
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
# Recreate the custom transformer used in training
class IQRCapper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        self.Q1_ = pd.DataFrame(X).quantile(0.25)
        self.Q3_ = pd.DataFrame(X).quantile(0.75)
        self.IQR_ = self.Q3_ - self.Q1_
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        lower_bound = self.Q1_ - self.factor * self.IQR_
        upper_bound = self.Q3_ + self.factor * self.IQR_
        return X.clip(lower=lower_bound, upper=upper_bound, axis=1)
# Load trained model
with open("Mobile_price_prediction_model.pkl", "rb") as f:
    model = pickle.load(f)


def predict_price(
    battery_power,
    has_bluetooth,
    clock_speed,
    dual_sim,
    front_camera,
    four_g,
    int_memory,
    m_depth,
    m_weight,
    n_cores,
    primary_camera,
    px_height,
    px_width,
    ram,
    talk_time,
    three_g,
    touch_screen,
    wifi
):

    # Automatically compute screen area
    screen_area = px_height * px_width

    data = pd.DataFrame({
        "battery_power":[battery_power],
        "has bluetooth":[has_bluetooth],
        "clock_speed":[clock_speed],
        "dual_sim":[dual_sim],
        "Front Camera px":[front_camera],
        "four_g":[four_g],
        "int_memory":[int_memory],
        "m_depth":[m_depth],
        "m_Weight":[m_weight],
        "n_cores":[n_cores],
        "Primary Camera px":[primary_camera],
        "px_height":[px_height],
        "px_width":[px_width],
        "ram":[ram],
        "talk_time":[talk_time],
        "three_g":[three_g],
        "touch_screen":[touch_screen],
        "wifi":[wifi],
        "screen_area":[screen_area]
    })

    prediction = model.predict(data)[0]

    labels = {
        0: "📉 Low Cost Mobile",
        1: "📊 Medium Cost Mobile",
        2: "📈 High Cost Mobile",
        3: "💎 Very High Cost Mobile"
    }

    return labels[prediction]


with gr.Blocks(title="Mobile Price Predictor") as demo:

    gr.Markdown("# 📱 Mobile Price Range Prediction")
    gr.Markdown("Enter mobile specifications to predict its price category.")

    with gr.Row():
        with gr.Column():

            battery_power = gr.Slider(500, 2000, label="Battery Power (mAh)")
            has_bluetooth = gr.Radio([0,1], label="Bluetooth (0=No, 1=Yes)")
            clock_speed = gr.Slider(0.5, 3.5, step=0.1, label="Clock Speed (GHz)")
            dual_sim = gr.Radio([0,1], label="Dual SIM")
            front_camera = gr.Slider(0, 20, label="Front Camera (MP)")
            primary_camera = gr.Slider(0, 50, label="Primary Camera (MP)")

            four_g = gr.Radio([0,1], label="4G Support")
            three_g = gr.Radio([0,1], label="3G Support")
            wifi = gr.Radio([0,1], label="WiFi")
            touch_screen = gr.Radio([0,1], label="Touch Screen")

        with gr.Column():

            ram = gr.Slider(256, 8000, step=128, label="RAM (MB)")
            int_memory = gr.Slider(2, 256, label="Internal Memory (GB)")
            n_cores = gr.Slider(1, 8, step=1, label="CPU Cores")
            m_weight = gr.Slider(80, 250, label="Mobile Weight (g)")
            m_depth = gr.Slider(0.1, 1.0, step=0.05, label="Mobile Depth")

            px_height = gr.Slider(0, 2000, label="Screen Pixel Height")
            px_width = gr.Slider(0, 2000, label="Screen Pixel Width")

            talk_time = gr.Slider(2, 30, label="Talk Time (hours)")

    predict_btn = gr.Button("🔮 Predict Price Range")

    output = gr.Textbox(label="Predicted Category")

    predict_btn.click(
        predict_price,
        inputs=[
            battery_power,
            has_bluetooth,
            clock_speed,
            dual_sim,
            front_camera,
            four_g,
            int_memory,
            m_depth,
            m_weight,
            n_cores,
            primary_camera,
            px_height,
            px_width,
            ram,
            talk_time,
            three_g,
            touch_screen,
            wifi
        ],
        outputs=output
    )


demo.launch()