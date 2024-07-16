const path = require('path');
const fs = require('fs');

const uploadDestination = 'D:/downloadsD/Data_Analyze_SQL-Packet/csv';

// Ensure the upload directory exists
if (!fs.existsSync(uploadDestination)) {
    try {
        fs.mkdirSync(uploadDestination, { recursive: true });
        console.log('Upload directory created:', uploadDestination);
    } catch (err) {
        console.error('Error creating upload directory:', err);
    }
}

const renameFile = (req, res) => {
    // Check if file is provided
    if (!req.file) {
        return res.status(400).send('No file uploaded.');
    }

    const tempPath = req.file.path;
    const targetPath = path.join(uploadDestination, 'test.csv');

    // Rename the file to test.csv
    fs.rename(tempPath, targetPath, (err) => {
        if (err) {
            return res.status(500).send('Error renaming the file.');
        }
        res.send('File uploaded and renamed to test.csv successfully.');
    });
};

const deleteFile = (req, res) => {
    const filePath = path.join(uploadDestination, 'test.csv');

    // Check if the file exists
    fs.access(filePath, fs.constants.F_OK, (err) => {
        if (err) {
            return res.status(404).send('File not found.');
        }

        // Delete the file
        fs.unlink(filePath, (err) => {
            if (err) {
                return res.status(500).send('Error deleting the file.');
            }
            res.send('File deleted successfully.');
        });
    });
};

module.exports = { renameFile, deleteFile };
