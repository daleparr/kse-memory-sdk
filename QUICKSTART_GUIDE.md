# 🚀 KSE Memory SDK - Quickstart Guide

**Get started with hybrid AI search in 30 seconds!**

## ⚡ Zero-Config Quick Start

```bash
# Install KSE Memory SDK
pip install kse-memory-sdk

# Run instant demo (auto-detects best backend)
kse quickstart

# That's it! 🎉
```

## 🎯 What Just Happened?

KSE Memory SDK automatically:
1. **Detected** the best available backend for your system
2. **Installed** any missing dependencies
3. **Configured** everything optimally
4. **Loaded** sample data and ran hybrid AI search
5. **Showed** you the power of Knowledge Graphs + Conceptual Spaces + Embeddings

## 🔧 Backend Options

### 🆓 **Free Options (No API Keys Required)**

#### ChromaDB (Recommended for Beginners)
```bash
kse quickstart --backend chromadb
```
- ✅ **Completely free forever**
- ✅ **Local data control**
- ✅ **Persistent storage**
- ✅ **No internet required**
- 🎯 **Perfect for**: Development, learning, small projects

#### Weaviate Cloud (Free Tier)
```bash
kse quickstart --backend weaviate
```
- ✅ **Free tier available**
- ✅ **Cloud managed**
- ✅ **Scalable**
- ⚠️ **Usage limits on free tier**
- 🎯 **Perfect for**: Prototypes, demos, small production

#### Qdrant Cloud (Free Tier)
```bash
kse quickstart --backend qdrant
```
- ✅ **High performance**
- ✅ **Free tier available**
- ✅ **Rust-based speed**
- ⚠️ **Usage limits on free tier**
- 🎯 **Perfect for**: Performance testing, ML workloads

### 💰 **Premium Options**

#### Pinecone (Production Ready)
```bash
# Set your API key first
export PINECONE_API_KEY="your-api-key"
kse quickstart --backend pinecone
```
- 🚀 **Enterprise grade**
- 🚀 **Unlimited scale**
- 🚀 **High availability**
- 💰 **Paid service**

## 🛠️ Interactive Setup

Want to compare all options and choose?

```bash
kse setup --interactive
```

This will:
1. Show you all available backends
2. Compare features, costs, and use cases
3. Help you choose the best option
4. Install and configure everything
5. Generate a production-ready config file

## 📋 Manual Configuration

### Step 1: Copy Environment Template
```bash
cp .env.example .env
```

### Step 2: Fill in Your API Keys (Optional)
```bash
# Edit .env file with your preferred editor
nano .env
```

### Step 3: Generate Configuration
```bash
kse setup --output my-config.yaml
```

### Step 4: Run with Custom Config
```bash
kse quickstart --config my-config.yaml
```

## 🎨 Demo Types

### Retail Demo (Default)
```bash
kse quickstart --demo-type retail
```
Experience product search with:
- Athletic footwear discovery
- Fashion recommendation
- Style-based filtering

### Finance Demo
```bash
kse quickstart --demo-type finance
```
Explore financial products:
- Investment opportunity matching
- Risk assessment tools
- Portfolio optimization

### Healthcare Demo
```bash
kse quickstart --demo-type healthcare
```
Discover medical solutions:
- Diagnostic equipment search
- Treatment protocol matching
- Medical device recommendations

## 🔍 Command Reference

### Basic Commands
```bash
# Auto-detect and run demo
kse quickstart

# Choose specific backend
kse quickstart --backend chromadb

# Run different demo type
kse quickstart --demo-type finance

# Skip web interface
kse quickstart --no-browser

# Save results to file
kse quickstart --output results.json
```

### Setup Commands
```bash
# Quick setup with auto-detection
kse setup

# Interactive setup with comparison
kse setup --interactive

# Save config to custom location
kse setup --output production.yaml
```

## 🌟 Next Steps

### For Developers
1. **Explore the Examples**: Check out `/examples/` directory
2. **Read the API Docs**: Understand the core concepts
3. **Build Your First App**: Use the generated config as a starting point

### For Data Scientists
1. **Try Different Backends**: Compare performance characteristics
2. **Experiment with Embeddings**: Test different models
3. **Analyze Search Results**: Understand hybrid scoring

### For DevOps/Production
1. **Review Security**: Check the production deployment guide
2. **Set Up Monitoring**: Configure health checks and metrics
3. **Scale Your Infrastructure**: Plan for growth

## 🆘 Troubleshooting

### "No backends detected"
```bash
# Install a free backend
pip install chromadb
kse quickstart --backend chromadb
```

### "API key required"
```bash
# Check your .env file
cat .env

# Or use a free backend instead
kse quickstart --backend chromadb
```

### "Connection failed"
```bash
# Check if services are running
docker ps

# Or use local backends
kse setup --interactive
```

## 🤝 Community & Support

- **GitHub**: [kse-memory/kse-memory-sdk](https://github.com/kse-memory/kse-memory-sdk)
- **Documentation**: [kse-memory-sdk.readthedocs.io](https://kse-memory-sdk.readthedocs.io)
- **Issues**: [Report bugs and request features](https://github.com/kse-memory/kse-memory-sdk/issues)
- **Discussions**: [Join the community](https://github.com/kse-memory/kse-memory-sdk/discussions)

## 🎯 Why KSE Memory?

### Traditional Search Problems
- **Keyword matching**: Misses semantic meaning
- **Vector search only**: Lacks structured knowledge
- **Complex setup**: Requires deep technical knowledge

### KSE Memory Solution
- **Hybrid AI**: Combines 3 complementary approaches
- **Zero configuration**: Works out of the box
- **Multiple backends**: Choose what fits your needs
- **Production ready**: Scales from laptop to enterprise

### The Magic Formula
```
KSE Search = Vector Embeddings + Knowledge Graphs + Conceptual Spaces
```

**Result**: More relevant, explainable, and intelligent search results.

---

**Ready to transform your search experience?**

```bash
pip install kse-memory-sdk && kse quickstart
```

🚀 **Welcome to the future of intelligent search!**