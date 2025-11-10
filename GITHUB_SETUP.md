# VulBERT GitHub Repository Setup Guide

## 📁 Essential Files for Deployment

Upload these files/folders to your GitHub repository for Streamlit Cloud deployment:

### ✅ REQUIRED FILES
```
📦 Your-Repository/
├── 📄 app.py                           # Main Streamlit webapp
├── 📄 requirements.txt                 # Python dependencies
├── 📄 .gitignore                       # Git ignore rules
├── 📄 DEPLOYMENT_README.md            # Deployment instructions
└── 📁 Models/                         # Model architecture classes
    ├── 📄 __init__.py
    ├── 📄 linevul_model.py
    └── 📄 StagedModel_line_vul.py
```

### ✅ REQUIRED FOLDERS
```
📦 Your-Repository/
├── 📁 Entry/                          # Training/inference scripts
│   ├── 📄 __init__.py
│   ├── 📄 linevul_main.py
│   └── 📄 StagedBert_line_vul.py
├── 📁 resource/                       # Models and tokenizers
│   ├── 📁 codebert-base/              # CodeBERT tokenizer (REQUIRED)
│   └── 📁 VulBERTa/                   # Base VulBERTa model
└── 📁 new_model/                      # Trained model checkpoints
    ├── 📁 vul/checkpoint-best-f1/     # Coarse model (REQUIRED)
    └── 📁 line_vul/checkpoint-best-f1/# Fine model (REQUIRED)
```

## 🚀 Deployment Steps

### 1. Create GitHub Repository
1. Go to GitHub.com
2. Click "New repository"
3. Name it (e.g., `vulbert-webapp`)
4. Make it public
5. Don't initialize with README

### 2. Upload Files
```bash
# Clone empty repo
git clone https://github.com/your-username/vulbert-webapp.git
cd vulbert-webapp

# Copy files from your local VulBERT folder
# (copy all files listed above)

# Add and commit
git add .
git commit -m "Initial commit: VulBERT webapp"
git push origin main
```

### 3. Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub account
3. Select your `vulbert-webapp` repository
4. Set main file path: `app.py`
5. Click "Deploy!"

## 📊 File Sizes & Requirements

- **Total size**: ~500MB-1GB (mainly model files)
- **Streamlit Cloud limits**: 1GB free tier, 3GB paid
- **Python version**: 3.8+
- **Memory**: Models load in ~2GB RAM

## 🔍 Verification Checklist

Before deploying, ensure:

- ✅ `app.py` imports without errors
- ✅ `requirements.txt` contains all dependencies
- ✅ Model files exist in correct paths:
  - `resource/codebert-base/` (tokenizer)
  - `new_model/vul/checkpoint-best-f1/` (coarse model)
  - `new_model/line_vul/checkpoint-best-f1/model_2048.bin` (fine model)
- ✅ All `__init__.py` files exist in subfolders

## 🐛 Troubleshooting

### Common Issues:
1. **"Model files not found"**: Check file paths in GitHub repo
2. **"Import errors"**: Verify `requirements.txt` is complete
3. **"Memory errors"**: Models need ~2GB RAM (free tier limit)

### Debug Commands:
```bash
# Test imports
python -c "import app; print('✅ Imports OK')"

# Check model paths
ls -la resource/codebert-base/
ls -la new_model/vul/checkpoint-best-f1/
ls -la new_model/line_vul/checkpoint-best-f1/
```

## 🎯 Final Result

After successful deployment, you'll have:
- 🌐 Public webapp URL (e.g., `https://your-app.streamlit.app`)
- 🛡️ Fully functional VulBERT vulnerability detector
- 📱 Responsive web interface
- ⚡ Real-time code analysis

## 📞 Support

If deployment fails:
1. Check Streamlit Cloud logs
2. Verify all required files are uploaded
3. Ensure file paths match exactly
4. Test locally first: `streamlit run app.py`

---

**🎉 Your VulBERT webapp is ready for the world!**</content>
<parameter name="filePath">c:\Users\Atharva Badgujar\Downloads\StagedVulBERT-master\StagedVulBERT-master\GITHUB_SETUP.md