# Queryfi: A Transfer Learning Approach to Semantic Parsing SQL Queries

Generating SQL queries from natural language has long been a popular and useful task attracting considerable interest. 
Recent years have seen a surge of large pre-trained language models demonstrating state-of-the-art performance across many natural language processing tasks. 
In this work, we explore leveraging two such models - BART and T5 - as well as traditional neural machine translation (NMT) architectures for the challenging problem of mapping natural language questions to structured SQL queries. 
While most prior approaches use recurrent or transformer architectures from scratch, we investigate transferring knowledge from these pre-trained seq2seq models and NMT systems to the text-to-SQL task through fine-tuning. Furthermore, the encoder-decoder framework of NMT models is well-suited to this cross-domain translation task. On the WikiSQL benchmark, our BART-based model outperforms previous best results, while the T5 and NMT models also achieve very competitive performance. We additionally propose a novel sequence-to-SQL generation framework leveraging BART to predict the different SQL clauses in parallel. 
Our analysis indicates BART, T5 and NMT models effectively capture SQL syntax and semantics, paving the way for pre-trained language models to generalize to other meaning representation tasks.
