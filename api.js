import express from "express";
import multer from "multer";
import fs from "fs";
import {
  generateOutput,
  generatePrompt,
  loadAndSplitTheDocs,
  vectorSaveAndSearch,
} from "./rag.js";

const PORT = 3005;

// Ensure the 'data' directory exists
const DATA_DIR = "data";
if (!fs.existsSync(DATA_DIR)) {
  fs.mkdirSync(DATA_DIR, { recursive: true });
  console.log(`Directory '${DATA_DIR}' created.`);
}

const upload = multer({
  storage: multer.diskStorage({
    destination: (req, file, cb) => {
      cb(null, DATA_DIR);
    },
    filename: (req, file, cb) => {
      const splittedFileName = file.originalname.split(".");
      const fileExtension = splittedFileName[splittedFileName.length - 1];
      const fileName = "sample." + fileExtension;
      cb(null, fileName);
    },
  }),
});

const app = express();
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.post("/upload", upload.single("file"), async (req, res) => {
  const {file} = req;
  const {question} = req.body;
  if (!file) {
    return res.status(400).send("No file uploaded.");
  }
  try {
    const splits = await loadAndSplitTheDocs(`./${DATA_DIR}/sample.pdf`);
    const searches = await vectorSaveAndSearch(splits, question); // get embeddings and search similarity
    const prompt = await generatePrompt(searches, question); // prompt pre processing
    const result = await generateOutput(prompt);
    res.json({
      message: "Content has been generated successfully.",
      data: {
        content: result.content,
      },
    });
  } catch (err) {
    console.error("Error processing file:", err);
    res.status(500).send("Internal Server Error");
  } finally {
    // Delete the uploaded file
    fs.unlink(`./${DATA_DIR}/sample.pdf`, (err) => {
      if (err) {
        console.error("Error deleting file:", err);
      } else {
        console.log("File deleted successfully.");
      }
    });
  }
});

app.listen(PORT, () => {
  console.log(`API is running on \nhttp://localhost:${PORT}`);
});
