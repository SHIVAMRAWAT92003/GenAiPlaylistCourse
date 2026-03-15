from langchain.text_splitter import RecursiveCharacterTextSplitter

text =""""

PyTorch is an open-source deep learning framework that has gained immense popularity among researchers and developers for its flexibility and ease of use. Built on Python and the Torch library, PyTorch offers dynamic computation graphs, which allow users to modify the network architecture during runtime, making it highly adaptable for experimentation and debugging. It supports GPU acceleration, enabling faster training of complex neural networks. PyTorch is widely used for building and training machine learning models, ranging from simple linear regressions to advanced deep learning architectures like convolutional and recurrent neural networks. Its intuitive interface, seamless integration with Python, and robust community support make it a preferred choice for both academic research and industry applications.


"""

spliter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap=0,
    
)

chunks = spliter.split_text(text)

print(len(chunks))
print(chunks)





