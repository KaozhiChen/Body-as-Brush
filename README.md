## Body-as-Brush: Minimal MVP

This is a minimal checkpoint implementation for the **Body-as-Brush** project.
It demonstrates:

- Real-time pose tracking using MediaPipe Pose.
- Using the right wrist as a virtual brush to draw light trails.
- Simple interaction controls via keyboard (as a placeholder for gesture-based commands).

### Installation

Create a virtual environment (optional but recommended), then install dependencies:

```bash
pip install -r requirements.txt
```

### Run the MVP

```bash
python -m src.body_as_brush
```

Make sure a webcam is connected and accessible.

### Controls

- **d**: Toggle drawing on/off.
- **c**: Clear the canvas.
- **1 / 2 / 3**: Change brush color.
- **q**: Quit the application.

