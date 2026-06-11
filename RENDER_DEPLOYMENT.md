# Render Deployment Guide

## Prerequisites
- GitHub account with your code pushed
- Render.com account (https://render.com)
- Model files (`model.h5`, `normalizer.npz`) in `saved_model/` directory

## Step-by-Step Deployment

### 1. Prepare Your Code
- Make sure all files are committed to GitHub
- The deployment files have been created:
  - `Procfile` - tells Render how to start your app
  - `render.yaml` - deployment configuration
  - `requirements.txt` - updated with `gunicorn` for production
  - `.gitignore` - to exclude unnecessary files

### 2. Create a Render Account
- Go to https://render.com
- Sign up with GitHub (recommended for easy integration)

### 3. Deploy Your App
1. **Login to Render Dashboard**
   - Go to https://dashboard.render.com

2. **Create New Web Service**
   - Click "+ New" button
   - Select "Web Service"

3. **Connect GitHub Repository**
   - Click "Connect a repository"
   - Search for your voice_sentiment_analysis repo
   - Click "Connect" next to it

4. **Configure Service**
   - **Name**: `voice-sentiment-analysis` (or your choice)
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: Free (for testing) or paid for production

5. **Add Environment Variables** (optional)
   - Click "Advanced" or go to Service Settings → Environment
   - Add if needed:
     ```
     FLASK_ENV=production
     ```

6. **Deploy**
   - Click "Create Web Service"
   - Render will automatically build and deploy your app
   - Wait for deployment to complete (2-3 minutes)

### 4. Access Your App
- Once deployed, you'll see a URL like: `https://voice-sentiment-analysis.onrender.com`
- Click the URL to access your app
- Default admin login: `admin` / `admin123`

### 5. Important Considerations

**Database:**
- SQLite database (`predictions.db`) will be created and stored locally
- Note: Files on Render's free tier are temporary and reset on app restart
- For persistent storage, upgrade to a paid plan or configure a PostgreSQL database

**Model Files:**
- Your trained model (`model.h5`, `normalizer.npz`) is in the repo
- These files must be committed to GitHub and will be deployed automatically

**File Uploads:**
- User-uploaded audio files are stored in `static/uploads/`
- Similarly, these are temporary on free tier
- Consider upgrading to persist files or use external storage (AWS S3)

**Memory & Performance:**
- Free tier has 0.5GB RAM - should be sufficient for audio inference
- Model loading may take a few seconds on first request (cold start)

### 6. Monitoring & Logs
- Go to your service dashboard on Render
- Click "Logs" to see real-time application output
- Check for any startup errors or issues

### 7. Custom Domain (Optional)
- In Service Settings → Custom Domains
- Add your own domain name
- Follow the DNS instructions

## Troubleshooting

**"Build failed"**
- Check logs for missing dependencies
- Ensure requirements.txt has all packages
- Verify Python syntax is correct

**"Application crashed"**
- Check logs for runtime errors
- Ensure model files exist in `saved_model/` directory
- Free tier may run out of memory - upgrade instance or optimize

**"Port binding failed"**
- The Procfile should have correct start command
- Render automatically assigns PORT environment variable

**"Files not persisting"**
- Free tier instances have temporary storage
- Upgrade to paid tier or use external storage service

## Next Steps
- Monitor app performance via dashboard
- Scale up to a paid plan if needed
- Set up automatic deployments on git push
- Consider adding a PostgreSQL database for persistent data

For more help: https://render.com/docs
