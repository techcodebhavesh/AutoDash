const express = require('express');
const cors = require('cors'); // Import CORS middleware
const SaveRouter = require('./routers/save.router.js');

const app = express();
const port = 5003;

// CORS middleware configuration
app.use(cors());

// Use the file routes
app.use('/files', SaveRouter);

// Start the server
app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});
