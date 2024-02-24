export async function getSubTableData() {
    const res = await fetch('http://127.0.0.1:8000/db_connect/sub_table/',  { cache: 'no-store' })
   
    if (!res.ok) {
      throw new Error('Failed to fetch sub_table data')
    }
   
    return res.json()
  }

// export async function getDataTableData({doc_id}) {

//     const res = await fetch(`http://127.0.0.1:8000/db_connect/data_table/${doc_id}/`)

//     if (!res.ok) {
//       throw new Error('Failed to fetch data_table data')
//     }
   
//     return res.json()
//   }

// export async function getDocument({bucket_name, file}) {

//     const res = await fetch(`http://127.0.0.1:8000/db_connect/get-document/${bucket_name}/${file}`)

//     if (!res.ok) {
//       throw new Error('Failed to fetch document from gcs')
//     }
   
//     return res

//   }





  