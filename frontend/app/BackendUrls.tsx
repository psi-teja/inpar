const hostIp = '127.0.0.1';

const BACKEND_URLS = {
  getSubTableUrl: `http://${hostIp}:8000/db_connect/get/SubTable`,
  uploadDocUrl: `http://${hostIp}:8000/db_connect/upload/doc/`,
  uploadGTJsonUrl: `http://${hostIp}:8000/db_connect/upload/json/gt_json`,
  getJsonUrl: `http://${hostIp}:8000/db_connect/get/json`,
  getDocUrl: `http://${hostIp}:8000/db_connect/get/document`,
  getOCRText: `http://${hostIp}:8000/db_connect/get/ocr_text`
};

export default BACKEND_URLS;
