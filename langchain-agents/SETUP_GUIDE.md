🚀 **GOOGLE CLOUD PROJECT SETUP GUIDE**
Project: gen-lang-client-0400019191 (899103612067)

## 🎯 **IMMEDIATE ACTION REQUIRED**

### 1. 📊 **ENABLE BILLING** (Most Critical)
**Link:** https://console.cloud.google.com/billing/linkedaccount?project=gen-lang-client-0400019191

**Steps:**
1. Click the link above
2. Select your project: `gen-lang-client-0400019191`
3. Click "LINK A BILLING ACCOUNT"
4. Add a credit card or payment method
5. ✅ Confirm billing is enabled

---

### 2. 🚀 **ENABLE GENERATIVE AI API**
**Link:** https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com?project=gen-lang-client-0400019191

**Steps:**
1. Click the link above (will auto-select your project)
2. Click the big blue **"ENABLE"** button
3. Wait for confirmation
4. ✅ API should show as "Enabled"

---

### 3. 🔑 **VERIFY API KEY**
**Link:** https://aistudio.google.com/app/apikey

**Steps:**
1. Click the link above
2. Ensure you're logged in with the same Google account
3. Check that your API key is associated with project `gen-lang-client-0400019191`
4. If not, create a new key for this project
5. ✅ Copy the new key to your .env file

---

### 4. 📋 **CHECK QUOTAS** (If still issues)
**Link:** https://console.cloud.google.com/iam-admin/quotas?project=gen-lang-client-0400019191

**Steps:**
1. Filter by: "Generative Language API"
2. Check quota limits
3. Request increases if needed

---

## 🧪 **TEST AFTER SETUP**

Run this command to test:
```bash
cd /Users/shashikantnanda/sunhacks/langchain-agents
uv run python test_enhanced.py
```

## 🎯 **EXPECTED OUTCOME**
- ✅ Billing enabled
- ✅ Generative Language API enabled  
- ✅ API key working
- ✅ Research system fully functional

## 🆘 **IF STILL NOT WORKING**
1. Try regenerating API key at: https://aistudio.google.com/app/apikey
2. Wait 5-10 minutes for changes to propagate
3. Ensure you're using the same Google account for everything

---
**Priority:** Complete steps 1 & 2 first - these fix 90% of issues!
