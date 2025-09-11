# ### Unsloth

# We're about to demonstrate the power of the new OpenAI GPT-OSS 20B model through an inference example. For our `bnb-4bit` version, use this [notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/GPT_OSS_BNB_(20B)-Inference.ipynb) instead.
# 
# **We're using OpenAI's MXFP4 Triton kernels combined with Unsloth's kernels!**

# In[2]:


from unsloth import FastLanguageModel
import torch

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/gpt-oss-20b-unsloth-bnb-4bit", # 20B model using bitsandbytes 4bit quantization
    "unsloth/gpt-oss-120b-unsloth-bnb-4bit",
    "unsloth/gpt-oss-20b", # 20B model using MXFP4 format
    "unsloth/gpt-oss-120b",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gpt-oss-20b",
    dtype = None, # None for auto detection
    max_seq_length = 4096, # Choose any for long context!
    load_in_4bit = False,  # 4 bit quantization to reduce memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)


# ### Reasoning Effort
# The `gpt-oss` models from OpenAI include a feature that allows users to adjust the model's "reasoning effort." This gives you control over the trade-off between the model's performance and its response speed (latency) which by the amount of token the model will use to think.
# 
# ----
# 
# The `gpt-oss` models offer three distinct levels of reasoning effort you can choose from:
# 
# * **Low**: Optimized for tasks that need very fast responses and don't require complex, multi-step reasoning.
# * **Medium**: A balance between performance and speed.
# * **High**: Provides the strongest reasoning performance for tasks that require it, though this results in higher latency.

# In[3]:


from transformers import TextStreamer

messages = [
    {"role": "user", "content": "Solve x^5 + 3x^4 - 10 = 3."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    return_tensors = "pt",
    return_dict = True,
    reasoning_effort = "low", # **NEW!** Set reasoning effort to low, medium or high
).to(model.device)

_ = model.generate(**inputs, max_new_tokens = 512, streamer = TextStreamer(tokenizer))


# Changing the `reasoning_effort` to `medium` will make the model think longer. We have to increase the `max_new_tokens` to occupy the amount of the generated tokens but it will give better and more correct answer

# In[4]:


from transformers import TextStreamer

messages = [
    {"role": "user", "content": "Solve x^5 + 3x^4 - 10 = 3."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    return_tensors = "pt",
    return_dict = True,
    reasoning_effort = "medium", # **NEW!** Set reasoning effort to low, medium or high
).to(model.device)

_ = model.generate(**inputs, max_new_tokens = 1024, streamer = TextStreamer(tokenizer))


# Lastly we will test it using `reasoning_effort` to `high`

# In[5]:


from transformers import TextStreamer

messages = [
    {"role": "user", "content": "Solve x^5 + 3x^4 - 10 = 3."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    return_tensors = "pt",
    return_dict = True,
    reasoning_effort = "high", # **NEW!** Set reasoning effort to low, medium or high
).to(model.device)

_ = model.generate(**inputs, max_new_tokens = 2048, streamer = TextStreamer(tokenizer))


# And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!
# 
# Some other links:
# 1. Train your own reasoning model - Llama GRPO notebook [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)
# 2. Saving finetunes to Ollama. [Free notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
# 3. Llama 3.2 Vision finetuning - Radiography use case. [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)
# 6. See notebooks for DPO, ORPO, Continued pretraining, conversational finetuning and more on our [documentation](https://docs.unsloth.ai/get-started/unsloth-notebooks)!
# 
# <div class="align-center">
#   <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
#   <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
#   <a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>
# 
#   Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️
# </div>
# 
