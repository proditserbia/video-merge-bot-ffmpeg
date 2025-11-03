# 🎬 Video Merge Bot --- FFmpeg Edition

![Video Merge Bot Interface](bot.JPG)

## 📍 Location & Launch

-   The bot (`MergeBot-FFMPEG.exe`) is installed in\
    **`C:\Portable\VideoMergeBot`** *(it can be placed anywhere on your
    system)*\
-   A shortcut is already on your **Desktop** for quick access.\
-   To start it, simply **double-click** the shortcut or the `.exe`
    file.\
-   FFmpeg and FFprobe are already added to your **system PATH**, so no
    additional setup is required.

------------------------------------------------------------------------

## ⚙️ Interface Overview

  -----------------------------------------------------------------------
  Field                   Description
  ----------------------- -----------------------------------------------
  **Base Folder**         Main directory where your source folders are
                          located. Each subfolder inside will be
                          processed individually.

  **Duration (min)**      Target duration for each output video. Example:
                          `90` = about 1h 15--30min total.

  **Outputs/Folder**      Number of output videos to generate per source
                          folder.

  **Daily Limit/Folder    Optional limit to control how many videos per
  (0=∞)**                 folder can be created daily. Set to `0` for no
                          limit.

  **Quality**             Output resolution (e.g., `2K` or `4K`).

  **Video Bitrate         Desired output bitrate. 20 Mb/s is a good
  (Mb/s)**                balance between quality and size.

  **Concurrency**         Number of parallel processes (GPU renders)
                          running at the same time. `2` is optimal for
                          RTX 4060.

  **ffmpeg / ffprobe      Normally left as `ffmpeg` and `ffprobe` since
  path**                  both are in your system PATH.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## ▶️ Basic Usage

1.  Click **Browse...** to select the main folder that contains your
    clip folders.\
2.  Adjust duration, bitrate, and concurrency if needed.\
3.  Press **START** to begin rendering.
    -   The bot will automatically skip broken or inconsistent clips.\
    -   Progress is logged live in the main window.\
4.  You can click **STOP (Kill All)** to abort all running processes
    immediately.\
5.  Click **FFmpeg Log** to open detailed logs for the current process.\
6.  Click **Show Stats** to view completed render statistics.

------------------------------------------------------------------------

## 💡 Notes

-   Each output video will be **at least 1 hour long** (typically 1h 15
    -- 1h 30 min).\
-   The bot supports folder names with **emojis** and non-ASCII
    characters.\
-   Optimal GPU utilization: **2 parallel renders ≈ 100 % GPU load** on
    RTX 4060.\
-   The bot can safely run **in the background** while you do other
    tasks.

------------------------------------------------------------------------

## 🧩 Manual Restart

If you ever need to restart the bot: 1. Close it completely (or use
**STOP → Kill All**).\
2. Double-click the shortcut again from your Desktop.\
3. The bot will resume with your last used settings.
