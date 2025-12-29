# ⚽ Real-Time Football Analytics with AI & Computer Vision

This project demonstrates a real-time football player detection, tracking, and performance analysis system using deep learning and computer vision technologies like Roboflow, BoT-SORT, OpenCV, and PyTorch.

## 🚀 Key Features

- 🎯 **Player Detection** using Roboflow API
- 🧠 **Object Tracking** via BoT-SORT
- 📏 **Speed Estimation** for individual players (m/s)
- 📍 **Distance Calculation** between players
- 📝 **Overlayed Video Output** with bounding boxes, IDs, and metrics
- 📊 **CSV Export** of all tracking data

## 🖥️ Technologies Used

- Python 3.x
- OpenCV
- PyTorch
- BoT-SORT (via Ultralytics)
- Roboflow API
- NumPy, Pandas, TQDM

## 📦 Requirements

Install the required libraries using pip:

```bash
pip install -r requirements.txt
```

## 📂 Project Structure

```
football-analytics/
├── football_analytics.py        # Main script
├── video.mp4                    # Input video (not included in repo)
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── football_analytics_output/   # Output folder (video + CSV)
```

## ▶️ How to Run

1. Place your input video as `video.mp4` in the project root.
2. Update your **Roboflow API key** and **model name** in the script if needed.
3. Run the script:

```bash
python football_analytics.py
```

- Press `q` to exit the live preview.
- Annotated video and CSV data will be saved in `football_analytics_output/`.

## 📈 Output

- `.mp4` annotated video with bounding boxes, speed, and IDs
- `.csv` file with player tracking data (frame, ID, position, speed)

## 📌 Notes

- Ensure your Roboflow project is configured with appropriate classes (e.g., players).
- Adjust `METERS_PER_PIXEL` to calibrate real-world scaling based on your video.

## 🤝 Credits

- [Roboflow](https://roboflow.com/)
- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [BoT-SORT](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)


🔗 Follow me on [GitHub](https://github.com/arafathosense) for more projects.
