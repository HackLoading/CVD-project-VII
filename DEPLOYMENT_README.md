# VulBERT - Code Vulnerability Detector 🚀

🛡️ **AI-Powered Code Security Analysis Tool**

VulBERT is a web application that uses machine learning to detect security vulnerabilities in code. It analyzes both function-level and line-level patterns to identify potential security issues like buffer overflows, use-after-free vulnerabilities, and other memory safety problems.

## 🌐 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_CLOUD_URL)

## ✨ Features

- **Function-Level Analysis**: Detects vulnerabilities at the function level using VulBERTa model
- **Line-Level Analysis**: Fine-grained analysis using LineVul model with pattern detection
- **Interactive Web Interface**: Clean, modern UI built with Streamlit
- **Real-time Analysis**: Instant vulnerability detection with confidence scores
- **Example Code Library**: Pre-loaded vulnerable and safe code examples
- **Debug Tools**: Model validation and debugging capabilities

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Backend**: PyTorch, Transformers
- **Models**: CodeBERT, VulBERTa, LineVul
- **Language**: Python 3.8+

## 🚀 Deployment on Streamlit Cloud

### Step 1: Prepare Your GitHub Repository

1. **Create a GitHub repository** for your VulBERT project
2. **Upload the following essential files**:
   ```
   vulbert/
   ├── app.py                    # Main Streamlit application
   ├── requirements.txt          # Python dependencies
   ├── Models/                   # Model classes
   │   ├── __init__.py
   │   ├── linevul_model.py
   │   └── StagedModel_line_vul.py
   ├── Entry/                    # Training/inference scripts
   │   ├── __init__.py
   │   ├── linevul_main.py
   │   └── StagedBert_line_vul.py
   ├── resource/                 # Models and tokenizers
   │   ├── codebert-base/        # CodeBERT tokenizer
   │   └── VulBERTa/            # Base VulBERTa model
   └── new_model/               # Trained model checkpoints
       ├── vul/checkpoint-best-f1/
       └── line_vul/checkpoint-best-f1/
   ```

### Step 2: Deploy on Streamlit Cloud

1. **Go to [Streamlit Cloud](https://streamlit.io/cloud)**
2. **Connect your GitHub account**
3. **Select your VulBERT repository**
4. **Configure deployment**:
   - **Main file path**: `app.py`
   - **Python version**: 3.8 or higher
5. **Deploy!** 🚀

### Step 3: Access Your App

Once deployed, you'll get a public URL like: `https://your-app-name.streamlit.app`

## 📋 Prerequisites

- Python 3.8 or higher
- Git

## 🏃‍♂️ Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/vulbert.git
   cd vulbert
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and go to `http://localhost:8501`

## 📊 Model Information

- **Coarse Model**: VulBERTa (Function-level detection)
- **Fine Model**: LineVul (Line-level detection)
- **Training Data**: BigVul Dataset (188K functions)
- **Accuracy**: ~94% on validation set
- **Supported Languages**: C, C++, Java (primarily)

## 🎯 Detection Capabilities

VulBERT can detect various types of vulnerabilities:

- **Buffer Overflows**: `strcpy()`, `gets()`, unbounded string operations
- **Use-After-Free**: Accessing memory after `free()`
- **Memory Safety Issues**: Double-free, uninitialized access
- **Format String Vulnerabilities**: Unsafe `printf()` usage
- **Integer Overflows**: Arithmetic operations that may overflow

## ⚠️ Important Limitations

- Analyzes **per-line**, not cross-line context
- Trained primarily on C/Java code
- Pattern-based detection (not semantic understanding)
- May produce false positives/negatives
- **Should NOT be used alone for production security decisions**

## 🔧 Configuration

### Vulnerability Threshold
Adjust the sensitivity of detection (0.0 to 1.0):
- **Low (0.2)**: More sensitive, may flag safe code
- **Medium (0.45)**: Balanced detection (default)
- **High (0.7)**: Conservative, may miss some vulnerabilities

### Example Code
Use the built-in examples to test the system:
- **Unsafe strcpy**: Classic buffer overflow
- **Safe strncpy**: Proper bounded copy
- **Use-after-free**: Memory safety violation
- **Integer Overflow**: Arithmetic overflow risk

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure all model files are uploaded to GitHub
   - Check file paths in `app.py`
   - Verify model file integrity

2. **Memory Issues**
   - Streamlit Cloud has memory limits
   - Models may take time to load on first run

3. **Import Errors**
   - Check `requirements.txt` has all dependencies
   - Ensure Python version compatibility

### Debug Mode
Enable debug tools in the sidebar to:
- Validate model accuracy
- Test model loading
- View detailed predictions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please cite appropriately if used in academic work.

## 🙏 Acknowledgments

- **BigVul Dataset**: Used for model training
- **CodeBERT**: Base model for code understanding
- **Hugging Face Transformers**: Model implementation
- **Streamlit**: Web application framework

## 📞 Support

For issues or questions:
- Open a GitHub issue
- Check the debug tools in the app
- Review the model validation results

---

**⚠️ Disclaimer**: This tool is for educational purposes only. Always perform manual code review and use multiple security analysis tools for production code.</content>
<parameter name="filePath">c:\Users\Atharva Badgujar\Downloads\StagedVulBERT-master\StagedVulBERT-master\DEPLOYMENT_README.md