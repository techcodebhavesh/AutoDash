const express = require('express');
const multer = require('multer');
const path = require('path');
const save = require('../controllers/save.controller.js');
const {
    renameFile,
    deleteFile
 
  } = require("../controllers/save.controller.js");

const Saverouter = express.Router();

// Set up multer for file uploads
const upload = multer({
    dest: path.join(__dirname, '../temp')
});

// Define the route for file upload
Saverouter.post('/upload', upload.single('file'), renameFile);

// Define the route for file deletion
Saverouter.delete('/delete', deleteFile);

module.exports = Saverouter;

    
