// db.js
import { Pool } from 'mysql2'; // or 'mysql2' for MySQL

const pool = new Pool({
  user: 'root',
  host: '35.238.38.191',
  database: 'invoiceparsing',
  password: '12345',
  port: 3306, // change it according to your database port
});

export default pool;
