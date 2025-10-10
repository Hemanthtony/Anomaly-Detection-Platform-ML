// Example of how to use Firebase services in your application
import { auth, db, storage } from './firebase-init.js';

// Example: Authentication
// Check if user is signed in
import { onAuthStateChanged } from "https://www.gstatic.com/firebasejs/9.22.2/firebase-auth.js";

onAuthStateChanged(auth, (user) => {
  if (user) {
    console.log('User is signed in:', user.email);
  } else {
    console.log('User is signed out');
  }
});

// Example: Firestore (Database)
// Save anomaly detection results
import { collection, addDoc, getDocs } from "https://www.gstatic.com/firebasejs/9.22.2/firebase-firestore.js";

async function saveAnomalyResult(anomalyData) {
  try {
    const docRef = await addDoc(collection(db, "anomalies"), anomalyData);
    console.log("Anomaly result saved with ID: ", docRef.id);
  } catch (e) {
    console.error("Error adding document: ", e);
  }
}

// Retrieve anomaly results
async function getAnomalyResults() {
  const querySnapshot = await getDocs(collection(db, "anomalies"));
  querySnapshot.forEach((doc) => {
    console.log(`${doc.id} => ${doc.data()}`);
  });
}

// Example: Storage
// Upload a file
import { ref, uploadBytes, getDownloadURL } from "https://www.gstatic.com/firebasejs/9.22.2/firebase-storage.js";

async function uploadFile(file) {
  const storageRef = ref(storage, 'anomaly-reports/' + file.name);
  try {
    const snapshot = await uploadBytes(storageRef, file);
    const downloadURL = await getDownloadURL(snapshot.ref);
    console.log('File uploaded successfully. Download URL:', downloadURL);
    return downloadURL;
  } catch (error) {
    console.error('Error uploading file:', error);
  }
}

// Usage examples:
// saveAnomalyResult({ type: 'network', timestamp: new Date(), severity: 'high' });
// getAnomalyResults();
// uploadFile(someFile);
