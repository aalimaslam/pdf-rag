import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { OllamaEmbeddings, ChatOllama } from "@langchain/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { PromptTemplate } from "@langchain/core/prompts";

export const loadAndSplitTheDocs = async (file_path) => {
  // load the uploaded file data
  const loader = new PDFLoader(file_path);
  const docs = await loader.load();
  console.log("Document Loaded")

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 0,
  });
  const allSplits = await textSplitter.splitDocuments(docs);
  console.log("Document Splitted")

  return allSplits;
};

export const vectorSaveAndSearch = async (splits,question) => {
    const embeddings = new OllamaEmbeddings();
    const vectorStore = await MemoryVectorStore.fromDocuments(
        splits,
        embeddings
    );

  console.log("Created Embedding")


    const searches = await vectorStore.similaritySearch(question);
    return searches;
};

export const generatePrompt = async (searches,question) =>
{
  console.log("Generating Prompt")

    let context = "";
    searches.forEach((search) => {
        context = context + "\n\n" + search.pageContent;
    });

    const prompt = PromptTemplate.fromTemplate(`

{context}

---

Answer the question based on the above context: {question}
`);

console.log("Generated Prompt")


    const formattedPrompt = await prompt.format({
        context: context,
        question: question,
    });

  console.log("Formatted Prompt")

    return formattedPrompt;
}


export const generateOutput = async (prompt) =>
{
  console.log("Generating Response")
    const ollamaLlm = new ChatOllama({
        baseUrl: "http://localhost:11434", // Default value
        model: "llama3.2", // Default value
    });

    const response = await ollamaLlm.invoke(prompt);
    
    console.log("Response Generated")
    return response;
}


