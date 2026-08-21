# User guide

## Image detection

1. Start the application with `streamlit run app.py`.
2. Open **Image detection** in the navigation.
3. Upload a readable JPG, JPEG, or PNG road image no larger than 10 MB.
4. Choose Fast, Balanced, or Thorough scanning. Balanced is recommended for difficult
   images; Thorough performs a second scale pass and takes longer.
5. Choose the minimum confidence. A higher value hides less-confident detections.
6. Select **Scan for potholes**.
7. Review the boxes, confidence values, count, timing, and apparent severity notice.
8. Download the annotated image. Save the summary only if you want it in local history.

If `models/best.pt` is missing, detection remains disabled and the page tells you where
to place the checkpoint.

## Video detection

Upload MP4, MOV, or AVI media no larger than 200 MB and no longer than five minutes.
Analyzing every second or third frame can improve speed but may miss brief detections.
The displayed unique count is an estimate, not a surveyed ground-truth count.

## Privacy

Uploaded media is processed locally. The image page keeps results only in the current
Streamlit session. The video page uses a uniquely named temporary directory and removes
its temporary files after reading the downloadable output into the session. History
stores only summaries unless storage behavior is deliberately extended.
