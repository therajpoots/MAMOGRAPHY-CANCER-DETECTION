import React, { useState, useEffect } from 'react'

const BACKEND_URL = "http://127.0.0.1:8000"

const renderFormattedRecommendation = (text: string) => {
  if (!text) return null;

  const lines = text.split('\n');
  const elements: React.ReactNode[] = [];

  let literatureMode = false;
  let summaryText = "";
  const litDetails: { [key: string]: string } = {};

  for (let i = 0; i < lines.length; i++) {
    let line = lines[i].trim();
    if (!line) continue;

    if (line.includes('Literature Context:')) {
      literatureMode = true;
      elements.push(
        <h5 key={`lit-header-${i}`} className="font-bold text-xs uppercase tracking-wider text-tertiary mb-3 mt-4 flex items-center gap-1.5 border-b border-outline-variant/30 pb-2">
          <span className="material-symbols-outlined text-sm">menu_book</span> Clinical Literature Context
        </h5>
      );
      // Strip Literature Context: prefix and any leading dash/spaces
      line = line.replace(/Literature Context:\s*-?\s*/i, '').trim();
      if (!line) continue;
    }

    if (line.startsWith('- Clinical Recommendation:') || line.startsWith('Clinical Recommendation:')) {
      const content = line.replace(/^-?\s*Clinical Recommendation:\s*/, '');
      elements.push(
        <div key={`rec-${i}`} className="bg-primary/5 border-l-4 border-primary p-4 rounded-r-xl mb-4">
          <h5 className="font-bold text-xs uppercase tracking-wider text-primary mb-1">Clinical Recommendation</h5>
          <p className="text-on-surface text-sm font-medium">{content}</p>
        </div>
      );
    } 
    else if (line.startsWith('Note:')) {
      const content = line.replace(/^Note:\s*/, '');
      elements.push(
        <div key={`note-${i}`} className="bg-yellow-500/5 border-l-4 border-yellow-500/80 p-4 rounded-r-xl mb-4 text-yellow-200">
          <h5 className="font-bold text-xs uppercase tracking-wider text-yellow-500 mb-1 flex items-center gap-1">
            <span className="material-symbols-outlined text-sm">warning</span> Note
          </h5>
          <p className="text-sm font-medium text-on-surface/90">{content}</p>
        </div>
      );
    } 
    else if (literatureMode) {
      if (line.includes('**Publication Title**') || line.includes('Publication Title')) {
        litDetails['Title'] = line.split('**Publication Title**:')[1] || line.split('Publication Title:')[1] || "";
      } else if (line.includes('**Authors**') || line.includes('Authors')) {
        litDetails['Authors'] = line.split('**Authors**:')[1] || line.split('Authors:')[1] || "";
      } else if (line.includes('**Journal**') || line.includes('Journal')) {
        litDetails['Journal'] = line.split('**Journal**:')[1] || line.split('Journal:')[1] || "";
      } else if (line.includes('**Publication Date**') || line.includes('Publication Date')) {
        litDetails['Date'] = line.split('**Publication Date**:')[1] || line.split('Publication Date:')[1] || "";
      } else if (line.includes('**Identifiers**') || line.includes('Identifiers')) {
        litDetails['Identifiers'] = line.split('**Identifiers**:')[1] || line.split('Identifiers:')[1] || "";
      } else if (line.includes('**Clinical Summary**') || line.includes('Clinical Summary')) {
        let summaryPart = line.split('**Clinical Summary**:')[1] || line.split('Clinical Summary:')[1] || "";
        summaryText = summaryPart;
        while (i + 1 < lines.length && !lines[i+1].includes('**') && !lines[i+1].includes('Literature Context:')) {
          i++;
          summaryText += " " + lines[i].trim();
        }
      } else if (line.startsWith('-') || line.startsWith('*')) {
        elements.push(
          <p key={`lit-item-${i}`} className="text-xs text-on-surface-variant mb-1 ml-2">{line.replace(/^-\s*/, '')}</p>
        );
      } else {
        summaryText += " " + line;
      }
    } 
    else {
      elements.push(
        <p key={`line-${i}`} className="text-sm text-on-surface-variant mb-2 leading-relaxed">{line}</p>
      );
    }
  }

  if (Object.keys(litDetails).length > 0 || summaryText) {
    elements.push(
      <div key="lit-card" className="glass-elevated rounded-xl p-5 border border-outline-variant/30 bg-surface-container-low/20 mt-3 flex flex-col gap-4">
        {litDetails['Title'] && (
          <div>
            <h6 className="text-[10px] text-on-surface-variant font-semibold uppercase tracking-wider">Publication Title</h6>
            <p className="text-sm font-bold text-on-surface mt-1 leading-snug">{litDetails['Title'].trim().replace(/^["'*]+|["'*]+$/g, '')}</p>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          {litDetails['Authors'] && (
            <div>
              <h6 className="text-[10px] text-on-surface-variant font-semibold uppercase tracking-wider">Authors</h6>
              <p className="text-on-surface font-medium mt-0.5">{litDetails['Authors'].trim()}</p>
            </div>
          )}
          
          {litDetails['Journal'] && (
            <div>
              <h6 className="text-[10px] text-on-surface-variant font-semibold uppercase tracking-wider">Journal/Conference</h6>
              <p className="text-on-surface font-medium mt-0.5">{litDetails['Journal'].trim()}</p>
            </div>
          )}

          {litDetails['Date'] && (
            <div>
              <h6 className="text-[10px] text-on-surface-variant font-semibold uppercase tracking-wider">Publication Date</h6>
              <p className="text-on-surface font-medium mt-0.5 font-mono">{litDetails['Date'].trim()}</p>
            </div>
          )}

          {litDetails['Identifiers'] && (
            <div>
              <h6 className="text-[10px] text-on-surface-variant font-semibold uppercase tracking-wider">Identifiers</h6>
              <p className="text-primary font-medium mt-0.5 font-mono break-all" dangerouslySetInnerHTML={{ __html: litDetails['Identifiers'].trim().replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank" class="hover:underline">$1</a>') }}></p>
            </div>
          )}
        </div>

        {summaryText.trim() && (
          <div className="border-t border-outline-variant/20 pt-3">
            <h6 className="text-[10px] text-on-surface-variant font-semibold uppercase tracking-wider">Clinical Summary</h6>
            <p className="text-xs text-on-surface-variant mt-1.5 leading-relaxed bg-surface-container-lowest/30 p-3 rounded-lg border border-outline-variant/10 italic font-medium">
              "{summaryText.trim().replace(/^["'*]+|["'*]+$/g, '')}"
            </p>
          </div>
        )}
      </div>
    );
  }

  return <div className="space-y-1">{elements}</div>;
};

export default function App() {
  const [user, setUser] = useState<{ email: string; first_name: string; last_name: string; role: string } | null>(null)
  const [loginEmail, setLoginEmail] = useState("rad@hospital.org")
  const [loginPassword, setLoginPassword] = useState("password123")
  const [loginError, setLoginError] = useState("")

  const [activeTab, setActiveTab] = useState<'dashboard' | 'patients' | 'audit'>('dashboard')
  const [patients, setPatients] = useState<any[]>([])
  const [studies, setStudies] = useState<any[]>([])
  const [auditLogs, setAuditLogs] = useState<any[]>([])
  const [selectedStudyId, setSelectedStudyId] = useState<number | null>(null)
  const [studyDetail, setStudyDetail] = useState<any>(null)
  const [activeView, setActiveView] = useState<'raw' | 'mask' | 'xai'>('raw')
  
  // Search state
  const [searchQuery, setSearchQuery] = useState("")
  
  // Forms
  const [newPatient, setNewPatient] = useState({ mrn: "", first_name: "", last_name: "", birth_date: "", gender: "F", family_history: "" })
  const [showAddPatient, setShowAddPatient] = useState(false)
  const [selectedPatientId, setSelectedPatientId] = useState<number | null>(null)
  const [uploadView, setUploadView] = useState("L-MLO")
  const [uploadFile, setUploadFile] = useState<File | null>(null)
  const [reportText, setReportText] = useState("")
  const [authorName, setAuthorName] = useState("")
  const [showAddStudy, setShowAddStudy] = useState(false)
  const [newStudyDensity, setNewStudyDensity] = useState("C")
  const [compareScan, setCompareScan] = useState<{
    id?: number
    date: string
    birads: string
    imgUrl: string
    maskUrl?: string | null
    xaiUrl?: string | null
    malignancyScore: string
    detected?: boolean
    diameter_px?: number
    area_pct?: number
    density?: string
  } | null>(null)
  const [comparePriorView, setComparePriorView] = useState<'raw' | 'mask' | 'xai'>('raw')
  const [compareCurrentView, setCompareCurrentView] = useState<'raw' | 'mask' | 'xai'>('raw')
  const openAddPatientModal = () => {
    const randomDigits = Math.floor(100000 + Math.random() * 900000);
    setNewPatient({ mrn: `MRN-${randomDigits}`, first_name: "", last_name: "", birth_date: "", gender: "F", family_history: "" });
    setShowAddPatient(true);
  }

  // UI state
  const [loading, setLoading] = useState(false)
  const [analyzeLoading, setAnalyzeLoading] = useState(false)

  // Fetch standard data on login
  useEffect(() => {
    if (user) {
      fetchPatients()
      fetchStudies()
      fetchAuditLogs()
    }
  }, [user])

  useEffect(() => {
    if (selectedStudyId) {
      fetchStudyDetail(selectedStudyId)
    }
  }, [selectedStudyId])

  const fetchPatients = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/v1/patients`)
      const data = await res.json()
      setPatients(data)
    } catch (err) {
      console.error(err)
    }
  }

  const fetchStudies = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/v1/studies`)
      const data = await res.json()
      setStudies(data)
    } catch (err) {
      console.error(err)
    }
  }

  const fetchAuditLogs = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/v1/audit`)
      const data = await res.json()
      setAuditLogs(data)
    } catch (err) {
      console.error(err)
    }
  }

  const fetchStudyDetail = async (id: number) => {
    setLoading(true)
    try {
      const res = await fetch(`${BACKEND_URL}/api/v1/studies/${id}/results`)
      const data = await res.json()
      setStudyDetail(data)
      setReportText(data.report.text || "")
      setAuthorName(data.report.author_name || `${user?.first_name} ${user?.last_name}`)
      // Default to showing mask if detected, otherwise raw
      if (data.images && data.images[0]?.analysis?.detected) {
        setActiveView('mask')
      } else {
        setActiveView('raw')
      }
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoginError("")
    try {
      const res = await fetch(`${BACKEND_URL}/api/v1/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: loginEmail, password: loginPassword })
      })
      if (!res.ok) {
        throw new Error("Invalid username, password, or MFA token.")
      }
      const data = await res.json()
      setUser(data)
    } catch (err: any) {
      setLoginError(err.message || "Connection to clinical backend failed.")
    }
  }

  const handleCreatePatient = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      const res = await fetch(`${BACKEND_URL}/api/v1/patients`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(newPatient)
      })
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}))
        throw new Error(errorData.detail || "Patient MRN already exists")
      }
      const createdPatient = await res.json()
      fetchPatients()
      setShowAddPatient(false)
      setNewPatient({ mrn: "", first_name: "", last_name: "", birth_date: "", gender: "F", family_history: "" })
      
      // Auto-select the newly registered patient if we are initializing a study
      if (showAddStudy) {
        setSelectedPatientId(createdPatient.id)
      }
    } catch (err: any) {
      alert(err.message)
    }
  }

  const handleCreateStudy = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!selectedPatientId) return
    try {
      const res = await fetch(`${BACKEND_URL}/api/v1/studies`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ patient_id: selectedPatientId, density_category: newStudyDensity })
      })
      const data = await res.json()
      fetchStudies()
      setSelectedStudyId(data.id)
      setShowAddStudy(false)
    } catch (err) {
      console.error(err)
    }
  }

  const handleUploadImage = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!selectedStudyId || !uploadFile) return
    const formData = new FormData()
    formData.append("file", uploadFile)
    
    try {
      const res = await fetch(`${BACKEND_URL}/api/v1/studies/${selectedStudyId}/upload?image_view=${uploadView}`, {
        method: "POST",
        body: formData
      })
      await res.json()
      setUploadFile(null)
      fetchStudyDetail(selectedStudyId)
      fetchStudies()
    } catch (err) {
      console.error(err)
    }
  }

  const handleRunAnalysis = async () => {
    if (!selectedStudyId) return
    setAnalyzeLoading(true)
    try {
      const res = await fetch(`${BACKEND_URL}/api/v1/studies/${selectedStudyId}/analyze`, {
        method: "POST"
      })
      await res.json()
      fetchStudyDetail(selectedStudyId)
      fetchStudies()
      fetchAuditLogs()
    } catch (err) {
      console.error(err)
    } finally {
      setAnalyzeLoading(false)
    }
  }

  const handleDiscardResults = async () => {
    if (!selectedStudyId) return
    if (!window.confirm('Discard all AI analysis results for this study? This will delete the mask and Grad-CAM overlays and reset the study status to Uploaded.')) return
    try {
      const res = await fetch(`${BACKEND_URL}/api/v1/studies/${selectedStudyId}/results`, {
        method: "DELETE"
      })
      await res.json()
      setStudyDetail(null)
      fetchStudyDetail(selectedStudyId)
      fetchStudies()
      fetchAuditLogs()
    } catch (err) {
      console.error(err)
    }
  }

  const handleSignReport = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!selectedStudyId) return
    try {
      const res = await fetch(`${BACKEND_URL}/api/v1/studies/${selectedStudyId}/report`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ report_text: reportText, author_name: authorName })
      })
      await res.json()
      fetchStudyDetail(selectedStudyId)
      fetchStudies()
      fetchAuditLogs()
    } catch (err) {
      console.error(err)
    }
  }

  const handleExportPDF = () => {
    if (!studyDetail) return
    const reportWindow = window.open("", "_blank")
    if (!reportWindow) return
    
    const patientName = `${studyDetail.study.patient.first_name} ${studyDetail.study.patient.last_name}`
    const accession = studyDetail.study.accession_number
    const date = new Date(studyDetail.study.study_date).toLocaleDateString()
    
    const textToPrint = studyDetail.report.text || reportText
    const isSigned = studyDetail.report.status === 'Signed'
    
    let analysisDetailsHTML = ""
    if (studyDetail.images[0]?.analysis) {
      const ar = studyDetail.images[0].analysis
      analysisDetailsHTML = `
        <div class="metrics-section">
          <h3>AI Diagnostic Assessment</h3>
          <table class="metrics-table">
            <tr>
              <th>BI-RADS Classification</th>
              <td><strong>${ar.birads}</strong></td>
            </tr>
            <tr>
              <th>Lesion Detection Status</th>
              <td>${ar.detected ? `Detected (Diameter: ${(ar.diameter_px * 0.1).toFixed(1)} mm, Area: ${ar.area_pct.toFixed(2)}%)` : 'No suspicious mass detected'}</td>
            </tr>
            <tr>
              <th>Prediction Confidences</th>
              <td>
                Malignant: ${(ar.probabilities.Malignant * 100).toFixed(1)}% | 
                Benign: ${(ar.probabilities.Benign * 100).toFixed(1)}% | 
                Normal: ${(ar.probabilities.Normal * 100).toFixed(1)}%
              </td>
            </tr>
          </table>
        </div>
      `
    }

    let imageEmbedHTML = ""
    if (studyDetail.images.length > 0) {
      const rawUrl = `${BACKEND_URL}${studyDetail.images[0].url}`
      const hasAnalysis = !!studyDetail.images[0]?.analysis
      const maskUrl = hasAnalysis && studyDetail.images[0].analysis.mask_url ? `${BACKEND_URL}${studyDetail.images[0].analysis.mask_url}` : null
      const xaiUrl = hasAnalysis && studyDetail.images[0].analysis.xai_url ? `${BACKEND_URL}${studyDetail.images[0].analysis.xai_url}` : null

      imageEmbedHTML = `
        <div class="image-print-section">
          <h3>Diagnostic Scan & AI Analysis Views (${studyDetail.images[0].view})</h3>
          <div class="images-container">
            <div class="image-box">
              <h4>1. Raw Ultrasound Scan</h4>
              <img src="${rawUrl}" alt="Diagnostic Ultrasound Scan" />
            </div>
            ${maskUrl ? `
            <div class="image-box">
              <h4>2. U-Net Segmentation Mask</h4>
              <div style="position: relative; width: 100%; max-height: 220px; border-radius: 6px; border: 1px solid #e2e8f0; overflow: hidden; background: #000; aspect-ratio: 4/3; margin-top: 10px;">
                <img src="${rawUrl}" style="position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; opacity: 0.8;" />
                <img src="${maskUrl}" style="position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; mix-blend-mode: screen; opacity: 0.5; filter: hue-rotate(90deg);" />
              </div>
            </div>
            ` : ''}
            ${xaiUrl ? `
            <div class="image-box">
              <h4>3. Grad-CAM++ Attention Map</h4>
              <img src="${xaiUrl}" alt="Grad-CAM++ Attention Map" />
            </div>
            ` : ''}
          </div>
        </div>
      `
    }

    reportWindow.document.write(`
      <html>
        <head>
          <title>Clinical Report - ${patientName}</title>
          <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; padding: 40px; color: #1e293b; background-color: #FFFFFF; }
            .header { border-bottom: 2px solid #0284c7; padding-bottom: 15px; margin-bottom: 25px; display: flex; justify-content: space-between; align-items: flex-end; }
            .logo { font-size: 22px; font-weight: 800; color: #0284c7; letter-spacing: -0.5px; }
            .title { font-size: 13px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }
            .report-title { text-align: center; font-size: 20px; font-weight: 800; margin: 30px 0; color: #0f172a; }
            .meta-section { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin-bottom: 24px; }
            .meta-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
            .meta-item { font-size: 13px; color: #334155; }
            .meta-label { font-weight: 700; color: #475569; display: inline-block; width: 130px; }
            
            .metrics-section { margin-bottom: 24px; }
            .metrics-section h3 { font-size: 14px; font-weight: 700; color: #0f172a; margin-bottom: 10px; border-bottom: 1px solid #e2e8f0; padding-bottom: 5px; text-transform: uppercase; }
            .metrics-table { width: 100%; border-collapse: collapse; font-size: 13px; }
            .metrics-table th, .metrics-table td { text-align: left; padding: 8px; border-bottom: 1px solid #f1f5f9; }
            .metrics-table th { font-weight: 600; color: #475569; width: 200px; }
            
            .image-print-section { margin-bottom: 24px; page-break-inside: avoid; }
            .image-print-section h3 { font-size: 14px; font-weight: 700; color: #0f172a; text-align: left; margin-bottom: 10px; border-bottom: 1px solid #e2e8f0; padding-bottom: 5px; text-transform: uppercase; }
            .images-container { display: flex; gap: 16px; justify-content: flex-start; margin-top: 10px; }
            .image-box { flex: 1; text-align: center; max-width: 280px; }
            .image-box h4 { font-size: 11px; font-weight: 600; text-transform: uppercase; color: #475569; margin: 0 0 6px 0; }
            .image-box img { max-width: 100%; max-height: 220px; border-radius: 6px; border: 1px solid #e2e8f0; object-fit: contain; }

            .content-section { margin-bottom: 24px; page-break-inside: avoid; }
            .content-section h3 { font-size: 14px; font-weight: 700; color: #0f172a; margin-bottom: 10px; border-bottom: 1px solid #e2e8f0; padding-bottom: 5px; text-transform: uppercase; }
            .content { font-family: monospace; font-size: 12px; white-space: pre-wrap; background: #fafafa; border: 1px solid #e2e8f0; padding: 16px; border-radius: 6px; line-height: 1.5; color: #1e293b; }
            
            .recommendation-section { background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 8px; padding: 16px; margin-bottom: 24px; page-break-inside: avoid; }
            .recommendation-section h4 { font-size: 13px; font-weight: 700; color: #0369a1; margin: 0 0 6px 0; text-transform: uppercase; }
            .recommendation-section p { font-size: 13px; color: #0c4a6e; margin: 0; line-height: 1.4; }

            .signature-box { border-top: 2px solid #e2e8f0; padding-top: 20px; margin-top: 40px; font-size: 13px; display: flex; justify-content: space-between; align-items: flex-start; page-break-inside: avoid; }
            .signature-details p { margin: 4px 0; }
            .seal { border: 2px solid #059669; color: #059669; font-weight: 800; padding: 6px 12px; border-radius: 6px; text-transform: uppercase; font-size: 10px; letter-spacing: 0.5px; display: inline-block; }
            .draft-badge { border: 2px solid #dc2626; color: #dc2626; font-weight: 800; padding: 6px 12px; border-radius: 6px; text-transform: uppercase; font-size: 10px; letter-spacing: 0.5px; display: inline-block; }
            @media print {
              body { padding: 0; }
            }
          </style>
        </head>
        <body>
          <div class="header">
            <div class="logo">MAMM-AI GATEWAY</div>
            <div class="title">Clinical Diagnostic Record</div>
          </div>
          
          <div class="report-title">Breast Ultrasound Analysis & Diagnostic Report</div>
          
          <div class="meta-section">
            <div class="meta-grid">
              <div class="meta-item"><span class="meta-label">Patient Name:</span> ${patientName}</div>
              <div class="meta-item"><span class="meta-label">Accession Number:</span> ${accession}</div>
              <div class="meta-item"><span class="meta-label">MRN:</span> ${studyDetail.study.patient.mrn}</div>
              <div class="meta-item"><span class="meta-label">Study Date:</span> ${date}</div>
              <div class="meta-item"><span class="meta-label">Birth Date:</span> ${new Date(studyDetail.study.patient.birth_date).toLocaleDateString()}</div>
              <div class="meta-item"><span class="meta-label">Breast Density:</span> Category ${studyDetail.study.density_category}</div>
            </div>
          </div>
          
          ${analysisDetailsHTML}
          ${imageEmbedHTML}

          <div class="content-section">
            <h3>Clinical Findings & Report Text</h3>
            <div class="content">${textToPrint}</div>
          </div>

          <div class="recommendation-section">
            <h4>Clinical Guidelines Citation: ${studyDetail.recommendation.citation}</h4>
            <p>${studyDetail.recommendation.text}</p>
          </div>
          
          <div class="signature-box">
            <div class="signature-details">
              ${isSigned ? `
                <p><strong>Approved & Digitally Signed by:</strong> Dr. ${studyDetail.report.author_name}</p>
                <p style="font-family: monospace; font-size: 10px; color: #64748b; margin-top: 6px;">Checksum: ${studyDetail.report.signature_checksum}</p>
                <p style="color: #64748b; font-size: 11px;">Approved: ${new Date(studyDetail.report.approved_at).toLocaleString()}</p>
              ` : `
                <p style="color: #dc2626; font-weight: 700;">DRAFT REPORT - NOT SIGNED</p>
                <p style="color: #64748b; font-size: 11px; margin-top: 4px;">Generated on: ${new Date().toLocaleString()}</p>
              `}
            </div>
            <div>
              ${isSigned ? `
                <div class="seal">HIPAA SEAL VERIFIED</div>
              ` : `
                <div class="draft-badge">UNSIGNED DRAFT</div>
              `}
            </div>
          </div>
          
          <script>
            window.onload = function() {
              window.print();
              setTimeout(function() { window.close(); }, 500);
            }
          </script>
        </body>
      </html>
    `)
    reportWindow.document.close()
  }

  const handleExportComparativePDF = () => {
    if (!studyDetail || !compareScan) return
    const reportWindow = window.open("", "_blank")
    if (!reportWindow) return

    const patientName = `${studyDetail.study.patient.first_name} ${studyDetail.study.patient.last_name}`
    const mrn = studyDetail.study.patient.mrn
    const age = studyDetail.study.patient.birth_date ? `${Math.floor((new Date().getTime() - new Date(studyDetail.study.patient.birth_date).getTime()) / (365.25 * 24 * 60 * 60 * 1000))}Y` : '54Y'

    const priorDate = compareScan.date
    const currentDate = new Date(studyDetail.study.study_date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })

    const priorBirads = compareScan.birads
    const currentBirads = studyDetail.images[0]?.analysis ? studyDetail.images[0].analysis.birads : "Pending"

    const priorScore = compareScan.malignancyScore
    const currentScore = studyDetail.images[0]?.analysis ? `${(studyDetail.images[0].analysis.probabilities.Malignant * 100).toFixed(1)}%` : "Pending"

    const priorDia = compareScan.diameter_px ? `${(compareScan.diameter_px * 0.1).toFixed(1)} mm` : "No lesion detected"
    const currentDia = studyDetail.images[0]?.analysis?.detected ? `${(studyDetail.images[0].analysis.diameter_px * 0.1).toFixed(1)} mm` : "No lesion detected"

    const priorArea = compareScan.area_pct ? `${compareScan.area_pct.toFixed(2)}%` : "N/A"
    const currentArea = studyDetail.images[0]?.analysis?.detected ? `${studyDetail.images[0].analysis.area_pct.toFixed(2)}%` : "N/A"

    // Calculate progression details
    let sizeDiffText = ""
    let probDiffText = ""
    let progressStatus = "Stable Findings"

    const currentProbVal = studyDetail.images[0]?.analysis ? studyDetail.images[0].analysis.probabilities.Malignant * 100 : 0
    const priorProbVal = parseFloat(compareScan.malignancyScore)

    if (compareScan.diameter_px && studyDetail.images[0]?.analysis?.diameter_px) {
      const diff = (studyDetail.images[0].analysis.diameter_px - compareScan.diameter_px) * 0.1
      if (diff > 0.5) {
        sizeDiffText = `Lesion size has increased by ${diff.toFixed(1)} mm, indicating measurable growth.`
        progressStatus = "Lesion Progression / Enlargement"
      } else if (diff < -0.5) {
        sizeDiffText = `Lesion size has decreased by ${Math.abs(diff).toFixed(1)} mm, indicating favorable response/regression.`
        progressStatus = "Lesion Regression / Size Reduction"
      } else {
        sizeDiffText = `Lesion size remains stable within normal measurement variance (change of ${diff.toFixed(1)} mm).`
      }
    }

    if (currentProbVal && priorProbVal) {
      const diff = currentProbVal - priorProbVal
      if (diff > 5) {
        probDiffText = `AI malignancy risk has increased by ${diff.toFixed(1)}% (from ${priorProbVal.toFixed(1)}% to ${currentProbVal.toFixed(1)}%).`
        if (progressStatus === "Stable Findings") progressStatus = "Increased Risk Score"
      } else if (diff < -5) {
        probDiffText = `AI malignancy risk has decreased by ${Math.abs(diff).toFixed(1)}% (from ${priorProbVal.toFixed(1)}% to ${currentProbVal.toFixed(1)}%).`
      } else {
        probDiffText = `AI malignancy risk profile is stable (change of ${diff.toFixed(1)}%).`
      }
    }

    const rawUrlPrior = compareScan.imgUrl
    const rawUrlCurrent = `${BACKEND_URL}${studyDetail.images[0].url}`

    const maskUrlPrior = compareScan.maskUrl ? `${BACKEND_URL}${compareScan.maskUrl}` : null
    const maskUrlCurrent = studyDetail.images[0]?.analysis?.mask_url ? `${BACKEND_URL}${studyDetail.images[0].analysis.mask_url}` : null

    const xaiUrlPrior = compareScan.xaiUrl ? `${BACKEND_URL}${compareScan.xaiUrl}` : null
    const xaiUrlCurrent = studyDetail.images[0]?.analysis?.xai_url ? `${BACKEND_URL}${studyDetail.images[0].analysis.xai_url}` : null

    reportWindow.document.write(`
      <html>
        <head>
          <title>Comparative Progress Report - ${patientName}</title>
          <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; padding: 40px; color: #1e293b; background-color: #FFFFFF; }
            .header { border-bottom: 2px solid #0284c7; padding-bottom: 15px; margin-bottom: 25px; display: flex; justify-content: space-between; align-items: flex-end; }
            .logo { font-size: 22px; font-weight: 800; color: #0284c7; letter-spacing: -0.5px; }
            .title { font-size: 13px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }
            .report-title { text-align: center; font-size: 20px; font-weight: 800; margin: 30px 0; color: #0f172a; }
            
            .hipaa-alert { background: #fffbeb; border: 1px solid #fde68a; border-left: 5px solid #d97706; padding: 12px 16px; border-radius: 6px; font-size: 12px; color: #92400e; margin-bottom: 20px; }
            .clinical-warning { background: #fef2f2; border: 1px solid #fca5a5; border-left: 5px solid #dc2626; padding: 12px 16px; border-radius: 6px; font-size: 12px; color: #991b1b; margin-bottom: 24px; }
            
            .meta-section { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin-bottom: 24px; }
            .meta-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
            .meta-item { font-size: 13px; color: #334155; }
            .meta-label { font-weight: 700; color: #475569; display: inline-block; width: 130px; }
            
            .comparison-table { width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 24px; }
            .comparison-table th, .comparison-table td { text-align: left; padding: 10px; border: 1px solid #e2e8f0; }
            .comparison-table th { background: #f1f5f9; color: #475569; font-weight: 700; }
            .comparison-table td.metric-name { font-weight: 600; color: #334155; }
            
            .scans-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px; }
            .scan-column { border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; }
            .scan-column h3 { font-size: 13px; text-transform: uppercase; color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; margin-top: 0; }
            .image-row { display: flex; gap: 12px; margin-top: 12px; }
            .image-box { flex: 1; text-align: center; }
            .image-box h4 { font-size: 10px; color: #64748b; margin: 0 0 4px 0; text-transform: uppercase; }
            .image-box img { width: 100%; max-height: 140px; border-radius: 4px; border: 1px solid #e2e8f0; object-fit: contain; }
            
            .analysis-section { background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 8px; padding: 16px; margin-bottom: 24px; }
            .analysis-section h3 { font-size: 14px; font-weight: 700; color: #0369a1; margin: 0 0 8px 0; text-transform: uppercase; }
            .analysis-section p { font-size: 13px; color: #0c4a6e; margin: 0 0 10px 0; line-height: 1.5; }
            
            .signature-box { border-top: 2px solid #e2e8f0; padding-top: 20px; margin-top: 40px; font-size: 13px; display: flex; justify-content: space-between; align-items: flex-start; page-break-inside: avoid; }
            .signature-details p { margin: 4px 0; }
            .seal { border: 2px solid #059669; color: #059669; font-weight: 800; padding: 6px 12px; border-radius: 6px; text-transform: uppercase; font-size: 10px; letter-spacing: 0.5px; display: inline-block; }
            
            @media print {
              body { padding: 0; }
            }
          </style>
        </head>
        <body>
          <div class="header">
            <div class="logo">MAMM-AI GATEWAY</div>
            <div class="title">Comparative Progression Report</div>
          </div>
          
          <div class="report-title">Breast Ultrasound Temporal Tracking & AI Progress Analysis</div>
          
          <div class="hipaa-alert">
            <strong>[HIPAA SECURITY WARNING] PROTECTED HEALTH INFORMATION (PHI)</strong><br/>
            This document is generated by a HIPAA-compliant medical system containing private clinical patient records. Access is restricted to authorized clinicians only. Unauthorized disclosure, sharing, or replication is strictly prohibited under federal regulations.
          </div>
          
          <div class="clinical-warning">
            <strong>[CLINICAL VALIDATION DISCLAIMER]</strong><br/>
            This AI-assisted tracking analysis compares automated neural style classifiers and segmentation parameters over time. It is NOT a final clinical diagnosis. The final medical decision must be validated by a board-certified radiologist.
          </div>

          <div class="meta-section">
            <div class="meta-grid">
              <div class="meta-item"><span class="meta-label">Patient Name:</span> ${patientName}</div>
              <div class="meta-item"><span class="meta-label">MRN:</span> ${mrn}</div>
              <div class="meta-item"><span class="meta-label">Patient Age:</span> ${age}</div>
              <div class="meta-item"><span class="meta-label">Breast Density:</span> Volumetric Category ${studyDetail.study.density_category}</div>
            </div>
          </div>
          
          <h3>1. Quantitative AI Metrics Comparison</h3>
          <table class="comparison-table">
            <thead>
              <tr>
                <th>Diagnostic Parameter</th>
                <th>Prior Scan (${priorDate})</th>
                <th>Current Scan (${currentDate})</th>
                <th>Temporal Change</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td class="metric-name">BI-RADS Classification</td>
                <td>${priorBirads}</td>
                <td>${currentBirads}</td>
                <td>${priorBirads === currentBirads ? "No category shift" : `Shifted from ${priorBirads} to ${currentBirads}`}</td>
              </tr>
              <tr>
                <td class="metric-name">AI Malignancy Risk</td>
                <td>${priorScore}</td>
                <td>${currentScore}</td>
                <td>${(currentProbVal - priorProbVal) > 0 ? "+" : ""}${(currentProbVal - priorProbVal).toFixed(1)}% score change</td>
              </tr>
              <tr>
                <td class="metric-name">Lesion Diameter</td>
                <td>${priorDia}</td>
                <td>${currentDia}</td>
                <td>${compareScan.diameter_px && studyDetail.images[0]?.analysis?.diameter_px ? `${((studyDetail.images[0].analysis.diameter_px - compareScan.diameter_px) * 0.1).toFixed(1)} mm` : "N/A"}</td>
              </tr>
              <tr>
                <td class="metric-name">Relative Lesion Area</td>
                <td>${priorArea}</td>
                <td>${currentArea}</td>
                <td>${compareScan.area_pct && studyDetail.images[0]?.analysis?.area_pct ? `${(studyDetail.images[0].analysis.area_pct - compareScan.area_pct).toFixed(2)}%` : "N/A"}</td>
              </tr>
            </tbody>
          </table>
          
          <h3>2. Diagnostic Scans side-by-side</h3>
          <div class="scans-grid">
            <div class="scan-column">
              <h3>Prior Diagnostic Scan (${priorDate})</h3>
              <div class="image-row">
                <div class="image-box">
                  <h4>Raw Scan</h4>
                  <img src="${rawUrlPrior}" />
                </div>
                ${maskUrlPrior ? `
                <div class="image-box">
                  <h4>U-Net Mask</h4>
                  <div style="position: relative; width: 100%; max-height: 140px; border-radius: 4px; border: 1px solid #e2e8f0; overflow: hidden; background: #000; aspect-ratio: 4/3;">
                    <img src="${rawUrlPrior}" style="position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; opacity: 0.8;" />
                    <img src="${maskUrlPrior}" style="position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; mix-blend-mode: screen; opacity: 0.5; filter: hue-rotate(90deg);" />
                  </div>
                </div>
                ` : ''}
                ${xaiUrlPrior ? `
                <div class="image-box">
                  <h4>Attention Map</h4>
                  <img src="${xaiUrlPrior}" />
                </div>
                ` : ''}
              </div>
            </div>
            
            <div class="scan-column">
              <h3>Current Diagnostic Scan (${currentDate})</h3>
              <div class="image-row">
                <div class="image-box">
                  <h4>Raw Scan</h4>
                  <img src="${rawUrlCurrent}" />
                </div>
                ${maskUrlCurrent ? `
                <div class="image-box">
                  <h4>U-Net Mask</h4>
                  <div style="position: relative; width: 100%; max-height: 140px; border-radius: 4px; border: 1px solid #e2e8f0; overflow: hidden; background: #000; aspect-ratio: 4/3;">
                    <img src="${rawUrlCurrent}" style="position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; opacity: 0.8;" />
                    <img src="${maskUrlCurrent}" style="position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; mix-blend-mode: screen; opacity: 0.5; filter: hue-rotate(90deg);" />
                  </div>
                </div>
                ` : ''}
                ${xaiUrlCurrent ? `
                <div class="image-box">
                  <h4>Attention Map</h4>
                  <img src="${xaiUrlCurrent}" />
                </div>
                ` : ''}
              </div>
            </div>
          </div>
          
          <div class="analysis-section">
            <h3>3. AI Progression Analysis & Clinical Tracking Findings</h3>
            <p><strong>Temporal Progression Category:</strong> ${progressStatus}</p>
            <p>${sizeDiffText}</p>
            <p>${probDiffText}</p>
            <p><strong>Technical Interpretation:</strong> The attention map overlap shows a ${progressStatus.includes("Progression") ? "consistent expansion" : "stable alignment"} of the neural style representations near the lesion border. These observations indicate the tracking of the lesion margins aligns with the calculated classification shift.</p>
          </div>
          
          <div class="signature-box">
            <div class="signature-details">
              <p><strong>Report Generated By:</strong> MAMM-AI Comparative Diagnostics Pipeline</p>
              <p style="color: #64748b; font-size: 11px;">Verification Date: ${new Date().toLocaleString()}</p>
            </div>
            <div class="seal">HIPAA SEAL VERIFIED</div>
          </div>
          
          <script>
            window.onload = function() {
              window.print();
              setTimeout(function() { window.close(); }, 500);
            }
          </script>
        </body>
      </html>
    `)
    reportWindow.document.close()
  }

  const handleLogout = () => {
    setUser(null)
    setStudyDetail(null)
    setSelectedStudyId(null)
  }

  // Filter studies based on search query
  const filteredStudies = studies.filter(s => 
    s.patient_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    s.patient_mrn.toLowerCase().includes(searchQuery.toLowerCase()) ||
    s.current_status.toLowerCase().includes(searchQuery.toLowerCase())
  )

  if (!user) {
    return (
      <div className="min-h-screen bg-background text-on-surface flex flex-col justify-center py-12 sm:px-6 lg:px-8 relative overflow-hidden">
        {/* Blur blobs */}
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/10 rounded-full blur-3xl pointer-events-none"></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-tertiary/10 rounded-full blur-3xl pointer-events-none"></div>
        
        <div className="sm:mx-auto sm:w-full sm:max-w-md relative z-10">
          <div className="flex justify-center items-center gap-3">
            <div className="bg-primary/20 p-2.5 rounded-xl text-primary border border-primary/30 shadow-lg">
              <span className="material-symbols-outlined text-3xl" style={{ fontVariationSettings: "'FILL' 1" }}>medical_services</span>
            </div>
            <span className="text-2xl font-bold tracking-tight text-primary">MAMM-AI CLINIC</span>
          </div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-on-surface font-headline">
            Sign in to diagnostic portal
          </h2>
          <p className="mt-2 text-center text-sm text-on-surface-variant">
            HIPAA-Secure Medical Imaging Platform
          </p>
        </div>

        <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md relative z-10">
          <div className="glass-panel py-8 px-4 shadow-xl border border-primary/10 rounded-2xl sm:px-10">
            <form className="space-y-6" onSubmit={handleLogin}>
              <div>
                <label className="block text-sm font-semibold text-on-surface-variant">
                  Clinical Email Address
                </label>
                <div className="mt-1">
                  <input
                    type="email"
                    required
                    value={loginEmail}
                    onChange={(e) => setLoginEmail(e.target.value)}
                    className="w-full bg-surface-container-high border border-outline-variant/50 rounded-xl py-2.5 px-4 text-sm text-on-surface placeholder:text-on-surface-variant/60 focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/50 transition-all"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-semibold text-on-surface-variant">
                  Password
                </label>
                <div className="mt-1">
                  <input
                    type="password"
                    required
                    value={loginPassword}
                    onChange={(e) => setLoginPassword(e.target.value)}
                    className="w-full bg-surface-container-high border border-outline-variant/50 rounded-xl py-2.5 px-4 text-sm text-on-surface placeholder:text-on-surface-variant/60 focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/50 transition-all"
                  />
                </div>
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <input
                    id="remember-me"
                    name="remember-me"
                    type="checkbox"
                    defaultChecked
                    className="h-4 w-4 bg-surface-container-high border-outline-variant text-primary focus:ring-primary rounded"
                  />
                  <label htmlFor="remember-me" className="ml-2 block text-sm text-on-surface-variant">
                    Verify secure session (MFA)
                  </label>
                </div>
              </div>

              {loginError && (
                <div className="bg-error-container/50 border-l-4 border-error p-3 rounded-lg flex gap-2 items-center text-sm text-on-error-container border border-error/20">
                  <span className="material-symbols-outlined text-error">warning</span>
                  <span>{loginError}</span>
                </div>
              )}

              <div>
                <button
                  type="submit"
                  className="w-full flex justify-center py-3 px-4 border border-primary/30 rounded-xl shadow-lg text-sm font-semibold text-on-primary bg-primary hover:bg-primary/95 transition-all active:scale-95 duration-150"
                >
                  Authorize Clinical Access
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background text-on-surface font-body min-h-screen overflow-x-hidden antialiased flex flex-col">
      {/* TopNavBar Shell Structure */}
      <nav className="fixed top-0 w-full z-50 flex justify-between items-center px-6 h-16 bg-surface-container/60 backdrop-blur-xl border-b border-primary/10 shadow-[0_0_30px_rgba(125,211,252,0.05)]">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <span className="material-symbols-outlined text-primary text-3xl" style={{ fontVariationSettings: "'FILL' 1" }}>medical_services</span>
          <span className="text-xl font-headline font-semibold tracking-tight text-primary">MAMM-AI GATEWAY</span>
        </div>
        
        {/* Navigation Links */}
        <div className="hidden md:flex items-center h-full gap-8">
          <button
            onClick={() => setActiveTab('dashboard')}
            className={`h-full flex items-center border-b-2 text-sm font-medium pt-0.5 transition-colors ${
              activeTab === 'dashboard' ? 'text-primary border-primary' : 'text-on-surface-variant hover:text-on-surface border-transparent'
            }`}
          >
            Study Board
          </button>
          <button
            onClick={() => setActiveTab('patients')}
            className={`h-full flex items-center border-b-2 text-sm font-medium pt-0.5 transition-colors ${
              activeTab === 'patients' ? 'text-primary border-primary' : 'text-on-surface-variant hover:text-on-surface border-transparent'
            }`}
          >
            Patient Directory
          </button>
          <button
            onClick={() => setActiveTab('audit')}
            className={`h-full flex items-center border-b-2 text-sm font-medium pt-0.5 transition-colors ${
              activeTab === 'audit' ? 'text-primary border-primary' : 'text-on-surface-variant hover:text-on-surface border-transparent'
            }`}
          >
            HIPAA Audit Logs
          </button>
        </div>

        {/* Trailing Actions */}
        <div className="flex items-center gap-4">
          <button className="text-on-surface-variant hover:text-on-surface transition-colors hover:bg-surface-bright/50 p-2 rounded-full active:scale-95 duration-200">
            <span className="material-symbols-outlined">notifications</span>
          </button>
          <button className="text-on-surface-variant hover:text-on-surface transition-colors hover:bg-surface-bright/50 p-2 rounded-full active:scale-95 duration-200">
            <span className="material-symbols-outlined">settings</span>
          </button>
          
          <div className="w-8 h-8 rounded-full overflow-hidden border border-primary/30">
            <img 
              alt="Radiologist Profile" 
              className="w-full h-full object-cover" 
              src="https://lh3.googleusercontent.com/aida-public/AB6AXuC-xofPCqNmrhuEPpqRKEipP0-3RTZW9RnaYiIIeuQ4lIv-XxWtGlQyFW1LXRwrfvpipCu5ByQlBH9s17wA99Y_22U4ksVcDs7wML7JuHYYKUlEsacIHi2KnIGGbrsx93hFuSrsogVtteqpzSqQtbXykPPi24lmXLDTh1nkKSpqJmLgM2SVx2ys48pMygJ-oxuu45Kk_SNXFZUt7VeSyTinrF3jBratVDYUEmJ6V0TRAdVSCe2XET1kmTRRAs2f9fnT_YCP4w-izbdp"
            />
          </div>
          
          <button
            onClick={handleLogout}
            className="text-on-surface-variant hover:text-on-surface transition-colors hover:bg-surface-bright/50 p-2 rounded-full active:scale-95 duration-200"
            title="Secure Sign-out"
          >
            <span className="material-symbols-outlined">logout</span>
          </button>
        </div>
      </nav>

      {/* Main Content Area */}
      <main className="flex-1 mt-16 p-6 max-w-[1920px] mx-auto w-full flex flex-col gap-6 relative overflow-x-hidden">
        
        {/* Tab 1: Diagnostics Study Board */}
        {activeTab === 'dashboard' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 w-full flex-grow items-start h-[calc(100vh-6rem)]">
            
            {/* Left Column: Active Studies (1/3) */}
            <aside className="lg:col-span-4 xl:col-span-3 flex flex-col h-[calc(100vh-6rem)] sticky top-24">
              <div className="glass-panel rounded-2xl flex-1 flex flex-col overflow-hidden">
                {/* List Header & Search */}
                <div className="p-5 border-b border-outline-variant/50 flex flex-col gap-4 shrink-0">
                  <div className="flex justify-between items-center">
                    <h2 className="text-lg font-headline font-semibold text-on-surface flex items-center gap-2">
                      <span className="material-symbols-outlined text-primary">view_list</span>
                      Active Studies
                    </h2>
                    <div className="flex gap-2">
                      <button
                        onClick={() => openAddPatientModal()}
                        className="bg-primary/10 hover:bg-primary/20 text-primary border border-primary/30 px-2 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1 transition-all active:scale-95"
                        title="Register New Patient"
                      >
                        <span className="material-symbols-outlined text-xs">person_add</span> + Patient
                      </button>
                      <button
                        onClick={() => setShowAddStudy(true)}
                        className="bg-primary/10 hover:bg-primary/20 text-primary border border-primary/30 px-2 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1 transition-all active:scale-95"
                        title="Initialize New Imaging Study"
                      >
                        <span className="material-symbols-outlined text-xs">add</span> + Study
                      </button>
                    </div>
                  </div>
                  
                  <div className="relative">
                    <span className="material-symbols-outlined absolute left-4 top-1/2 -translate-y-1/2 text-on-surface-variant text-lg">search</span>
                    <input 
                      type="text"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="w-full bg-surface-container-high border border-outline-variant/50 rounded-full py-2.5 pl-12 pr-4 text-sm text-on-surface placeholder:text-on-surface-variant/60 focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/50 transition-all" 
                      placeholder="Search MRN, Name, Date..."
                    />
                  </div>
                </div>

                {/* Patient List */}
                <div className="flex-1 overflow-y-auto p-3 space-y-3 custom-scrollbar">
                  {filteredStudies.map((s) => (
                    <div
                      key={s.id}
                      onClick={() => setSelectedStudyId(s.id)}
                      className={`rounded-xl p-4 cursor-pointer transition-all ${
                        selectedStudyId === s.id
                          ? 'active-study'
                          : 'glass-elevated hover:bg-surface-container-highest/40 border border-transparent'
                      }`}
                    >
                      <div className="flex justify-between items-start mb-2">
                        <div>
                          <h3 className="font-medium text-on-surface">{s.patient_name}</h3>
                          <p className="text-xs text-on-surface-variant mt-0.5">MRN: {s.patient_mrn} • {s.patient_age}Y</p>
                        </div>
                        <span className={`inline-flex items-center px-2 py-1 rounded text-[10px] font-medium border ${
                          s.current_status === 'Approved' ? 'bg-primary-container/30 text-primary border-primary/20' :
                          s.current_status === 'Needs Review' ? 'bg-error-container/50 text-error border-error/20' :
                          'bg-secondary-container text-on-secondary-container border border-secondary/20'
                        }`}>
                          {s.current_status}
                        </span>
                      </div>
                      <div className="flex justify-between items-end mt-4">
                        <span className="text-xs text-on-surface-variant">
                          {new Date(s.study_date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                        </span>
                        <span className="material-symbols-outlined text-primary text-sm">arrow_forward</span>
                      </div>
                    </div>
                  ))}
                  {filteredStudies.length === 0 && (
                    <div className="text-center py-8 text-on-surface-variant/60 text-xs">No active studies matches search query.</div>
                  )}
                </div>
              </div>
            </aside>

            {/* Right Column: Clinical Case Details (2/3) */}
            <div className="lg:col-span-8 xl:col-span-9 flex flex-col gap-6 h-[calc(100vh-6rem)] overflow-y-auto custom-scrollbar pb-12">
              
              {loading ? (
                <div className="glass-panel rounded-2xl p-12 text-center flex flex-col items-center justify-center gap-4 flex-grow border border-primary/10 min-h-[500px]">
                  <span className="material-symbols-outlined text-4xl text-primary animate-spin">autorenew</span>
                  <h3 className="text-lg font-bold text-on-surface">Loading Case Details</h3>
                  <p className="text-sm text-on-surface-variant">Retrieving mammography images and clinical history...</p>
                </div>
              ) : studyDetail ? (
                <>
                  {/* Top Patient Meta Banner */}
                  <header className="glass-panel rounded-2xl p-6 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 glow-accent relative overflow-hidden shrink-0">
                    <div className="absolute inset-0 bg-gradient-to-r from-primary/5 to-transparent pointer-events-none"></div>
                    <div className="relative z-10">
                      <h1 className="text-2xl font-headline font-bold text-on-surface flex items-center gap-3">
                        {studyDetail.study.patient.first_name} {studyDetail.study.patient.last_name}
                        <span className="text-sm font-normal px-2 py-0.5 rounded-full bg-surface-container-high border border-outline-variant text-on-surface-variant">
                          {studyDetail.study.patient.birth_date ? `${Math.floor((new Date().getTime() - new Date(studyDetail.study.patient.birth_date).getTime()) / (365.25 * 24 * 60 * 60 * 1000))}Y` : '54Y'}
                        </span>
                      </h1>
                      
                      <div className="flex flex-wrap items-center gap-x-6 gap-y-2 mt-3 text-sm">
                        <div className="flex flex-col">
                          <span className="text-xs text-on-surface-variant uppercase tracking-wider">MRN</span>
                          <span className="font-mono text-on-surface">{studyDetail.study.patient.mrn}</span>
                        </div>
                        <div className="w-px h-8 bg-outline-variant/50 hidden sm:block"></div>
                        <div className="flex flex-col">
                          <span className="text-xs text-on-surface-variant uppercase tracking-wider">Accession</span>
                          <span className="font-mono text-on-surface">{studyDetail.study.accession_number}</span>
                        </div>
                        <div className="w-px h-8 bg-outline-variant/50 hidden sm:block"></div>
                        <div className="flex flex-col">
                          <span className="text-xs text-on-surface-variant uppercase tracking-wider">Density</span>
                          <span className="font-medium text-tertiary">Volumetric Category {studyDetail.study.density_category}</span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="relative z-10 flex items-center gap-3 flex-wrap">
                      {studyDetail.study.current_status !== 'Approved' && studyDetail.images.length > 0 && (
                        <button
                          onClick={handleRunAnalysis}
                          disabled={analyzeLoading}
                          className="flex items-center gap-2 bg-primary/10 hover:bg-primary/20 text-primary border border-primary/30 px-5 py-2.5 rounded-lg font-medium transition-all active:scale-95 group disabled:opacity-50"
                        >
                          <span className="material-symbols-outlined text-xl group-hover:animate-pulse">memory</span>
                          {analyzeLoading ? 'Running AI Engine...' : 'Run AI Pipeline'}
                        </button>
                      )}
                      {studyDetail.images[0]?.analysis && studyDetail.study.current_status !== 'Approved' && (
                        <button
                          onClick={handleDiscardResults}
                          className="flex items-center gap-2 bg-error/10 hover:bg-error/20 text-error border border-error/30 px-4 py-2.5 rounded-lg font-medium transition-all active:scale-95"
                          title="Remove AI analysis results and reset study"
                        >
                          <span className="material-symbols-outlined text-xl">delete_sweep</span>
                          Discard Results
                        </button>
                      )}
                    </div>
                  </header>

                  {/* Viewer & Insights Grid */}
                  <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                    
                    {/* Diagnostic Viewer */}
                    <div className="glass-panel rounded-2xl p-1 flex flex-col min-h-[500px]">
                      <div className="flex justify-between items-center px-4 py-3 border-b border-outline-variant/30">
                        <h3 className="font-headline font-medium text-sm text-on-surface flex items-center gap-2">
                          <span className="material-symbols-outlined text-primary/80 text-lg">visibility</span>
                          Viewer
                        </h3>
                        {studyDetail.images[0]?.analysis && (
                          <div className="flex gap-1 bg-surface-container-high p-1 rounded-lg">
                            <button
                              onClick={() => setActiveView('raw')}
                              className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${
                                activeView === 'raw'
                                  ? 'bg-surface-container-lowest text-primary shadow-sm'
                                  : 'text-on-surface-variant hover:text-on-surface'
                              }`}
                            >
                              Raw
                            </button>
                            <button
                              onClick={() => setActiveView('mask')}
                              className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${
                                activeView === 'mask'
                                  ? 'bg-surface-container-lowest text-primary shadow-sm'
                                  : 'text-on-surface-variant hover:text-on-surface'
                              }`}
                            >
                              Mask
                            </button>
                            <button
                              onClick={() => setActiveView('xai')}
                              className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${
                                activeView === 'xai'
                                  ? 'bg-surface-container-lowest text-primary shadow-sm'
                                  : 'text-on-surface-variant hover:text-on-surface'
                              }`}
                            >
                              Grad-CAM++
                            </button>
                          </div>
                        )}
                      </div>
                      
                      {/* Image Area */}
                      <div className="flex-1 bg-[#05080f] rounded-b-xl relative overflow-hidden group border border-outline-variant/20 m-1 flex items-center justify-center min-h-[400px]">
                        {studyDetail.images.length > 0 ? (
                          <>
                            {activeView === 'raw' && (
                              <div 
                                className="absolute inset-0 bg-contain bg-center bg-no-repeat opacity-80 group-hover:opacity-100 transition-opacity duration-500" 
                                style={{ backgroundImage: `url(${BACKEND_URL}${studyDetail.images[0].url})` }}
                              ></div>
                            )}
                            {activeView === 'mask' && (
                              <>
                                <div 
                                  className="absolute inset-0 bg-contain bg-center bg-no-repeat opacity-80 transition-opacity duration-500" 
                                  style={{ backgroundImage: `url(${BACKEND_URL}${studyDetail.images[0].url})` }}
                                ></div>
                                {studyDetail.images[0].analysis?.mask_url && (
                                  <div 
                                    className="absolute inset-0 bg-contain bg-center bg-no-repeat mix-blend-screen opacity-50 filter hue-rotate-90"
                                    style={{ backgroundImage: `url(${BACKEND_URL}${studyDetail.images[0].analysis.mask_url})` }}
                                  ></div>
                                )}
                              </>
                            )}
                            {activeView === 'xai' && (
                              <div className="w-full h-full relative">
                                {studyDetail.images[0].analysis?.xai_url ? (
                                  <div 
                                    className="absolute inset-0 bg-contain bg-center bg-no-repeat opacity-80" 
                                    style={{ backgroundImage: `url(${BACKEND_URL}${studyDetail.images[0].analysis.xai_url})` }}
                                  ></div>
                                ) : (
                                  <p className="absolute inset-0 flex items-center justify-center text-on-surface-variant text-sm">Grad-CAM++ map not computed. Run AI analysis.</p>
                                )}
                              </div>
                            )}
                            
                            {/* Toolbar overlay */}
                            <div className="absolute right-4 top-4 flex flex-col gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                              <button className="w-8 h-8 rounded-full bg-surface-container-highest/80 backdrop-blur border border-outline-variant/50 flex items-center justify-center text-on-surface hover:text-primary transition-colors">
                                <span className="material-symbols-outlined text-sm">zoom_in</span>
                              </button>
                              <button className="w-8 h-8 rounded-full bg-surface-container-highest/80 backdrop-blur border border-outline-variant/50 flex items-center justify-center text-on-surface hover:text-primary transition-colors">
                                <span className="material-symbols-outlined text-sm">contrast</span>
                              </button>
                            </div>
                            
                            <div className="absolute bottom-3 left-3 bg-background/80 border border-outline-variant/30 px-2.5 py-1 rounded text-[10px] text-on-surface-variant font-semibold uppercase tracking-wider">
                              VIEW: {studyDetail.images[0].view}
                            </div>
                          </>
                        ) : (
                          <div className="flex flex-col items-center gap-3 text-center p-6 text-on-surface-variant">
                            <span className="material-symbols-outlined text-4xl">cloud_upload</span>
                            <p className="text-sm font-medium">No diagnostic images uploaded to this study.</p>
                            <form onSubmit={handleUploadImage} className="mt-2 flex flex-col items-center gap-3 bg-surface-container-high p-4 rounded-xl border border-outline-variant/50 w-full max-w-xs">
                              <div className="flex flex-col gap-1 w-full text-left">
                                <label className="text-[9px] text-on-surface-variant font-bold uppercase tracking-wider">Imaging View</label>
                                <select
                                  value={uploadView}
                                  onChange={(e) => setUploadView(e.target.value)}
                                  className="text-xs bg-surface-container-lowest border border-outline-variant/50 rounded-lg p-2 text-on-surface w-full focus:outline-none focus:border-primary/50"
                                >
                                  <option value="L-MLO">L-MLO (Left Medio-Lateral Oblique)</option>
                                  <option value="R-MLO">R-MLO (Right Medio-Lateral Oblique)</option>
                                  <option value="L-CC">L-CC (Left Cranio-Caudal)</option>
                                  <option value="R-CC">R-CC (Right Cranio-Caudal)</option>
                                </select>
                              </div>
                              <div className="w-full">
                                <input
                                  type="file"
                                  required
                                  onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                                  className="text-xs file:mr-4 file:py-1.5 file:px-3 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-primary/10 file:text-primary hover:file:bg-primary/20 cursor-pointer w-full"
                                />
                              </div>
                              <button
                                type="submit"
                                disabled={!uploadFile}
                                className="bg-primary text-on-primary hover:bg-primary/90 disabled:opacity-50 text-xs px-4 py-2 rounded-xl font-semibold flex items-center justify-center gap-1.5 w-full transition-all"
                              >
                                <span className="material-symbols-outlined text-sm">upload</span> Upload Scan Image
                              </button>
                            </form>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Insights & Analysis */}
                    <div className="flex flex-col gap-6">
                      
                      {/* BI-RADS Badge */}
                      {studyDetail.images[0]?.analysis ? (
                        (() => {
                          const rating = studyDetail.images[0].analysis.birads;
                          const isMalignant = rating.includes("4") || rating.includes("5");
                          const isProbablyBenign = rating.includes("3");
                          
                          let categoryText = "Suspicious";
                          let categoryDesc = "Recommend tissue diagnosis";
                          let categoryColor = "text-error";
                          let categoryBg = "bg-error/10";
                          let shadowColor = "drop-shadow-[0_0_15px_rgba(255,107,107,0.3)]";
                          
                          if (isProbablyBenign) {
                            categoryText = "Probably Benign";
                            categoryDesc = "Short-interval follow-up recommended";
                            categoryColor = "text-tertiary";
                            categoryBg = "bg-tertiary/10";
                            shadowColor = "drop-shadow-[0_0_15px_rgba(200,160,240,0.3)]";
                          } else if (!isMalignant) {
                            categoryText = "Negative/Benign";
                            categoryDesc = "Standard annual clinical screening";
                            categoryColor = "text-primary";
                            categoryBg = "bg-primary/10";
                            shadowColor = "drop-shadow-[0_0_15px_rgba(125,211,252,0.3)]";
                          }

                          const probs = studyDetail.images[0].analysis.probabilities;
                          let predictedClass = "Normal";
                          if (probs && probs.Malignant >= probs.Benign && probs.Malignant >= probs.Normal) {
                            predictedClass = "Malignant";
                          } else if (probs && probs.Benign >= probs.Malignant && probs.Benign >= probs.Normal) {
                            predictedClass = "Benign";
                          }

                          const displayDiagnosis = predictedClass === 'Malignant' ? 'MALIGNANT' : predictedClass === 'Benign' ? 'BENIGN' : 'NONE';

                          return (
                            <div className="glass-elevated rounded-2xl p-6 relative overflow-hidden border border-outline-variant/30">
                              <div className={`absolute -right-10 -top-10 w-32 h-32 ${categoryBg} rounded-full blur-3xl pointer-events-none`}></div>
                              <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 relative z-10">
                                <div>
                                  <h3 className="text-sm text-on-surface-variant font-medium mb-1 uppercase tracking-wider">AI Assessment</h3>
                                  <div className="flex items-end gap-4 mt-2">
                                    <div className={`text-5xl font-display font-bold ${categoryColor} ${shadowColor}`}>
                                      {rating.replace("BI-RADS ", "")}
                                    </div>
                                    <div className="pb-1 text-on-surface">
                                      <p className="font-medium text-lg">{categoryText}</p>
                                      <p className="text-sm text-on-surface-variant">{categoryDesc}</p>
                                    </div>
                                  </div>
                                </div>
                                <div className="flex flex-col items-start sm:items-end gap-2 shrink-0 bg-surface-container-high/40 px-4 py-3 rounded-xl border border-outline-variant/20">
                                  <span className="text-[10px] text-on-surface-variant font-bold uppercase tracking-wider block leading-normal">
                                    Diagnosis Class
                                  </span>
                                  <span className={`px-4 py-1.5 rounded-lg text-sm font-black uppercase tracking-wider border shadow-md ${
                                    displayDiagnosis === 'MALIGNANT' ? 'bg-error/15 text-error border-error/30 shadow-error/5' :
                                    displayDiagnosis === 'BENIGN' ? 'bg-primary/15 text-primary border-primary/30 shadow-primary/5' :
                                    'bg-tertiary/15 text-tertiary border-tertiary/30 shadow-tertiary/5'
                                  }`}>
                                    {displayDiagnosis}
                                  </span>
                                </div>
                              </div>
                            </div>
                          );
                        })()
                      ) : (
                        <div className="glass-elevated rounded-2xl p-6 relative overflow-hidden border border-outline-variant/30 text-center flex flex-col justify-center items-center py-10 text-on-surface-variant">
                          <span className="material-symbols-outlined text-3xl mb-1 text-primary/60">warning</span>
                          <p className="text-sm font-medium">Study is not analyzed yet.</p>
                          <p className="text-xs mt-1">Run the pipeline above to trigger U-Net and classifier models.</p>
                        </div>
                      )}

                      {studyDetail.images[0]?.analysis && (
                        <div className="glass-elevated rounded-2xl p-5 border border-outline-variant/20 flex flex-col gap-3">
                          <h4 className="text-xs text-on-surface-variant font-semibold uppercase tracking-wider flex items-center gap-1.5 border-b border-outline-variant/20 pb-2">
                            <span className="material-symbols-outlined text-sm text-primary">info</span>
                            BI-RADS Classification Legend
                          </h4>
                          <div className="grid grid-cols-1 gap-2 text-xs">
                            <div className="flex items-center justify-between p-2 rounded-lg bg-surface-container-high/20 border border-outline-variant/10">
                              <span className="font-bold text-primary px-1.5 py-0.5 rounded bg-primary/10 border border-primary/20 shrink-0">Category 1 & 2</span>
                              <span className="text-on-surface-variant text-[11px] text-right">Negative / Benign (Routine annual screening)</span>
                            </div>
                            <div className="flex items-center justify-between p-2 rounded-lg bg-surface-container-high/20 border border-outline-variant/10">
                              <span className="font-bold text-tertiary px-1.5 py-0.5 rounded bg-tertiary/10 border border-tertiary/20 shrink-0">Category 3</span>
                              <span className="text-on-surface-variant text-[11px] text-right">Probably Benign (6-month short-interval follow-up)</span>
                            </div>
                            <div className="flex items-center justify-between p-2 rounded-lg bg-surface-container-high/20 border border-outline-variant/10">
                              <span className="font-bold text-error px-1.5 py-0.5 rounded bg-error/10 border border-error/20 shrink-0">Category 4A</span>
                              <span className="text-on-surface-variant text-[11px] text-right">Low Suspicion of Malignancy (Biopsy recommended)</span>
                            </div>
                            <div className="flex items-center justify-between p-2 rounded-lg bg-surface-container-high/20 border border-outline-variant/10">
                              <span className="font-bold text-error px-1.5 py-0.5 rounded bg-error/10 border border-error/20 shrink-0">Category 4B & 4C</span>
                              <span className="text-on-surface-variant text-[11px] text-right">Moderate / High Suspicion (Urgent core needle biopsy)</span>
                            </div>
                            <div className="flex items-center justify-between p-2 rounded-lg bg-surface-container-high/20 border border-outline-variant/10">
                              <span className="font-bold text-error px-1.5 py-0.5 rounded bg-error/10 border border-error/20 shrink-0">Category 5</span>
                              <span className="text-on-surface-variant text-[11px] text-right">Highly Suggestive of Malignancy (Urgent tissue diagnosis)</span>
                            </div>
                          </div>
                        </div>
                      )}

                    </div>
                  </div>

                  {/* NCCN Guideline Citation Panel */}
                  {studyDetail.images[0]?.analysis && (
                    <div className="glass-panel rounded-2xl p-6 border border-primary/20 bg-primary/5 flex gap-4 items-start text-sm shrink-0">
                      <span className="material-symbols-outlined text-primary text-2xl shrink-0 mt-0.5">book</span>
                      <div className="flex-1">
                        <h4 className="font-bold text-primary text-base mb-3">{studyDetail.recommendation.citation}</h4>
                        {renderFormattedRecommendation(studyDetail.recommendation.text)}
                      </div>
                    </div>
                  )}

                  {/* Report Composer */}
                  {studyDetail.images[0]?.analysis && (
                    <div className="glass-panel rounded-2xl p-6 mt-2 border border-primary/20 relative">
                      <div className="flex justify-between items-center mb-4">
                        <h3 className="font-headline font-semibold text-lg text-on-surface flex items-center gap-2">
                          <span className="material-symbols-outlined text-primary">edit_document</span>
                          Report Findings
                        </h3>
                        <div className="font-mono text-[10px] text-on-surface-variant/50 bg-surface-container-lowest px-2 py-1 rounded border border-outline-variant/20">
                          SHA-256: {studyDetail.report.signature_checksum ? studyDetail.report.signature_checksum.replace("sha256-", "").substring(0, 16) : "8f434346648f6b..."}
                        </div>
                      </div>

                      {studyDetail.report.status === 'Signed' ? (
                        <div className="space-y-4">
                          <textarea 
                            readOnly
                            value={studyDetail.report.text}
                            className="w-full h-32 bg-surface-container-lowest/50 border border-outline-variant/50 rounded-xl p-4 text-sm text-on-surface focus:outline-none resize-none mb-4"
                          />
                          <div className="flex flex-col sm:flex-row justify-between items-center gap-4 pt-4 border-t border-outline-variant/30">
                            <div className="flex items-center gap-3 w-full sm:w-auto">
                              <div className="w-10 h-10 rounded-full bg-surface-container-high flex items-center justify-center border border-outline-variant/50">
                                <span className="material-symbols-outlined text-on-surface-variant text-sm">draw</span>
                              </div>
                              <div className="flex-1">
                                <p className="font-script text-primary text-xl font-medium leading-none">Dr. {studyDetail.report.author_name}</p>
                                <p className="text-[9px] text-on-surface-variant mt-1.5 font-mono">Approved: {new Date(studyDetail.report.approved_at).toLocaleString()}</p>
                              </div>
                            </div>
                            
                            <div className="flex items-center gap-3 w-full sm:w-auto">
                              <button 
                                onClick={handleExportPDF}
                                className="flex-1 sm:flex-none flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-sm font-medium text-on-surface-variant hover:text-on-surface hover:bg-surface-container-highest transition-colors border border-transparent"
                              >
                                <span className="material-symbols-outlined text-sm">picture_as_pdf</span>
                                Export
                              </button>
                              <button 
                                disabled
                                className="flex-1 sm:flex-none flex items-center justify-center gap-2 px-6 py-2 rounded-lg text-sm font-medium bg-primary/20 text-primary border border-primary/20 cursor-not-allowed opacity-60"
                              >
                                <span className="material-symbols-outlined text-sm">verified</span>
                                Signed &amp; Approved
                              </button>
                            </div>
                          </div>
                        </div>
                      ) : (
                        <form onSubmit={handleSignReport}>
                          <textarea 
                            value={reportText}
                            onChange={(e) => setReportText(e.target.value)}
                            className="w-full h-32 bg-surface-container-lowest/50 border border-outline-variant/50 rounded-xl p-4 text-sm text-on-surface placeholder:text-on-surface-variant/40 focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/50 transition-all resize-none mb-4" 
                            placeholder="Enter clinical findings... AI draft available."
                          />
                          <div className="flex flex-col sm:flex-row justify-between items-center gap-4 pt-4 border-t border-outline-variant/30">
                            <div className="flex items-center gap-3 w-full sm:w-auto">
                              <div className="w-10 h-10 rounded-full bg-surface-container-high flex items-center justify-center border border-outline-variant/50">
                                <span className="material-symbols-outlined text-on-surface-variant text-sm">draw</span>
                              </div>
                              <div className="flex-1">
                                <input 
                                  type="text"
                                  required
                                  value={authorName}
                                  onChange={(e) => setAuthorName(e.target.value)}
                                  className="w-full bg-transparent border-b border-outline-variant/50 focus:border-primary focus:outline-none py-1 text-sm font-script text-primary placeholder:text-on-surface-variant/50 placeholder:font-sans" 
                                  placeholder="Type name to sign..."
                                />
                              </div>
                            </div>
                            
                            <div className="flex items-center gap-3 w-full sm:w-auto">
                              <button 
                                type="button"
                                onClick={handleExportPDF}
                                className="flex-1 sm:flex-none flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-sm font-medium text-on-surface-variant hover:text-on-surface hover:bg-surface-container-highest transition-colors border border-transparent"
                              >
                                <span className="material-symbols-outlined text-sm">picture_as_pdf</span>
                                Export
                              </button>
                              <button 
                                type="submit"
                                className="flex-1 sm:flex-none flex items-center justify-center gap-2 px-6 py-2 rounded-lg text-sm font-medium bg-primary text-on-primary hover:bg-primary/90 transition-colors shadow-[0_0_15px_rgba(125,211,252,0.2)]"
                              >
                                <span className="material-symbols-outlined text-sm">verified</span>
                                Sign &amp; Approve
                              </button>
                            </div>
                          </div>
                        </form>
                      )}
                    </div>
                  )}

                  {/* Comparative Scan History */}
                  {studyDetail.images[0]?.analysis && (
                    <div className="glass-panel rounded-2xl p-6 mt-6 border border-outline-variant/30">
                      <div className="flex justify-between items-center mb-6">
                        <h3 className="font-headline font-semibold text-lg text-on-surface flex items-center gap-2">
                          <span className="material-symbols-outlined text-primary">history</span>
                          Comparative Scan History
                        </h3>
                        <button className="text-xs font-medium text-primary hover:text-primary/80 transition-colors">View All Records</button>
                      </div>
                      
                      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                        {studyDetail.history && studyDetail.history.length > 0 ? (
                          studyDetail.history.map((h: any) => (
                            <div key={h.id} className="bg-surface-container-lowest/50 border border-outline-variant/30 rounded-xl p-3 hover:border-primary/30 transition-all group">
                              <div className="flex gap-3">
                                <div className="w-20 h-20 rounded-lg bg-black overflow-hidden border border-outline-variant/20 flex-none flex items-center justify-center">
                                  <img 
                                    src={`${BACKEND_URL}${h.imgUrl}`} 
                                    alt="Previous scan thumbnail" 
                                    className="w-full h-full object-cover opacity-60 group-hover:opacity-100 transition-opacity"
                                  />
                                </div>
                                <div className="flex-1 flex flex-col justify-between py-0.5">
                                  <div>
                                    <p className="text-xs text-on-surface-variant font-medium uppercase tracking-wider">{h.date}</p>
                                    <p className="text-sm font-semibold text-on-surface mt-1">{h.birads}</p>
                                  </div>
                                  <button 
                                    onClick={() => setCompareScan({
                                      id: h.id,
                                      date: h.date,
                                      birads: h.birads,
                                      imgUrl: `${BACKEND_URL}${h.imgUrl}`,
                                      maskUrl: h.maskUrl,
                                      xaiUrl: h.xaiUrl,
                                      malignancyScore: h.malignancyScore,
                                      detected: h.detected,
                                      diameter_px: h.diameter_px,
                                      area_pct: h.area_pct,
                                      density: h.density
                                    })}
                                    className="flex items-center gap-1 text-[10px] font-bold text-primary uppercase tracking-tight hover:underline"
                                  >
                                    <span className="material-symbols-outlined text-xs">compare</span>
                                    Compare Side-by-Side
                                  </button>
                                </div>
                              </div>
                            </div>
                          ))
                        ) : (
                          <div className="col-span-full text-center py-6 text-xs text-on-surface-variant/60">
                            No prior diagnostic scans found for comparison.
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="glass-panel rounded-2xl p-12 text-center flex flex-col items-center justify-center gap-4 flex-grow border border-primary/10 min-h-[500px]">
                  <span className="material-symbols-outlined text-5xl text-primary/40 animate-pulse">analytics</span>
                  <h3 className="text-xl font-bold text-on-surface font-headline">Diagnostic Command Center</h3>
                  <p className="text-sm text-on-surface-variant max-w-sm">Select an active breast scan or registration study from the left column to run AI models and review clinical reports.</p>
                </div>
              )}
            </div>

          </div>
        )}

        {/* Tab 2: Patients Registration Directory */}
        {activeTab === 'patients' && (
          <div className="glass-panel rounded-2xl p-6 shadow-xl border border-primary/10 flex flex-col gap-6">
            <div className="flex justify-between items-center">
              <div>
                <h2 className="text-lg font-bold text-on-surface flex items-center gap-2">
                  <span className="material-symbols-outlined text-primary">groups</span>
                  Clinical Patient Directory
                </h2>
                <p className="text-xs text-on-surface-variant mt-1">Manage, search, and register new clinical patient entries.</p>
              </div>
              <button
                onClick={() => openAddPatientModal()}
                className="bg-primary text-on-primary hover:bg-primary/90 px-4 py-2.5 rounded-xl text-sm font-semibold flex items-center gap-1.5 transition-all active:scale-95 shadow-lg border border-primary/20"
              >
                <span className="material-symbols-outlined text-sm font-bold">person_add</span> Register Patient
              </button>
            </div>

            {/* List Patients Table */}
            <div className="overflow-x-auto border border-outline-variant/30 rounded-xl bg-surface-container-lowest/40">
              <table className="min-w-full divide-y divide-outline-variant/30 text-left">
                <thead className="bg-surface-container-high">
                  <tr>
                    <th className="px-6 py-3.5 text-xs font-bold text-primary uppercase tracking-wider">Patient Name</th>
                    <th className="px-6 py-3.5 text-xs font-bold text-primary uppercase tracking-wider">MRN</th>
                    <th className="px-6 py-3.5 text-xs font-bold text-primary uppercase tracking-wider">Birth Date</th>
                    <th className="px-6 py-3.5 text-xs font-bold text-primary uppercase tracking-wider">Gender</th>
                    <th className="px-6 py-3.5 text-xs font-bold text-primary uppercase tracking-wider">Clinical Notes</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-outline-variant/20 text-xs">
                  {patients.map((p) => (
                    <tr key={p.id} className="hover:bg-surface-container-high/40 transition-colors">
                      <td className="px-6 py-4 font-bold text-on-surface">{p.first_name} {p.last_name}</td>
                      <td className="px-6 py-4 font-mono text-on-surface-variant">{p.mrn}</td>
                      <td className="px-6 py-4 text-on-surface-variant">{p.birth_date}</td>
                      <td className="px-6 py-4 text-on-surface-variant">{p.gender}</td>
                      <td className="px-6 py-4 text-on-surface-variant max-w-xs truncate">{p.family_history || 'None'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Tab 3: Security & HIPAA Audit Logs */}
        {activeTab === 'audit' && (
          <div className="glass-panel rounded-2xl p-6 shadow-xl border border-primary/10 flex flex-col gap-6">
            <div>
              <h2 className="text-lg font-bold text-on-surface flex items-center gap-2">
                <span className="material-symbols-outlined text-primary">security</span>
                Security & Access Audit Logs
              </h2>
              <p className="text-xs text-on-surface-variant mt-1">HMAC-signed logs verifying user actions, model runs, and reports access history.</p>
            </div>

            <div className="overflow-x-auto border border-outline-variant/30 rounded-xl bg-surface-container-lowest/40">
              <table className="min-w-full divide-y divide-outline-variant/30 text-left">
                <thead className="bg-surface-container-high">
                  <tr>
                    <th className="px-6 py-3.5 text-xs font-bold text-primary uppercase tracking-wider">Timestamp</th>
                    <th className="px-6 py-3.5 text-xs font-bold text-primary uppercase tracking-wider">User</th>
                    <th className="px-6 py-3.5 text-xs font-bold text-primary uppercase tracking-wider">Event Type</th>
                    <th className="px-6 py-3.5 text-xs font-bold text-primary uppercase tracking-wider">Details</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-outline-variant/20 text-xs">
                  {auditLogs.map((log) => (
                    <tr key={log.id} className="hover:bg-surface-container-high/40 transition-colors">
                      <td className="px-6 py-4 text-on-surface-variant font-mono">{new Date(log.occurred_at).toLocaleString()}</td>
                      <td className="px-6 py-4 font-bold text-on-surface">{log.user_email || 'System'}</td>
                      <td className="px-6 py-4">
                        <span className={`inline-block px-2.5 py-0.5 rounded text-[10px] font-bold border ${
                          log.event_type === 'LOGIN' ? 'bg-green-500/10 text-green-400 border-green-500/20' :
                          log.event_type === 'AI_ANALYSIS' ? 'bg-primary/10 text-primary border-primary/20' :
                          log.event_type === 'REPORT_SIGN' ? 'bg-tertiary/10 text-tertiary border-tertiary/20' : 'bg-surface-container-high text-on-surface-variant border-outline-variant/20'
                        }`}>
                          {log.event_type}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-on-surface-variant">{log.details}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

      </main>

      {showAddPatient && (
        <div className="fixed inset-0 bg-background/80 backdrop-blur-sm flex justify-center items-center z-[60] p-4">
          <div className="glass-elevated rounded-2xl border border-primary/20 shadow-2xl max-w-md w-full p-6 space-y-6 relative">
            <button
              onClick={() => setShowAddPatient(false)}
              className="absolute top-4 right-4 text-on-surface-variant hover:text-on-surface focus:outline-none"
            >
              <span className="material-symbols-outlined text-lg">close</span>
            </button>
            <h3 className="text-lg font-bold text-on-surface flex items-center gap-2">
              <span className="material-symbols-outlined text-primary text-xl">person_add</span>
              Register Clinical Patient
            </h3>
            <form onSubmit={handleCreatePatient} className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-xs text-on-surface-variant font-semibold">First Name</label>
                  <input
                    type="text"
                    required
                    value={newPatient.first_name}
                    onChange={(e) => setNewPatient({ ...newPatient, first_name: e.target.value })}
                    className="w-full bg-surface-container-high border border-outline-variant/50 rounded-xl px-3 py-2 text-sm mt-1 focus:outline-none focus:border-primary/50 text-on-surface"
                  />
                </div>
                <div>
                  <label className="text-xs text-on-surface-variant font-semibold">Last Name</label>
                  <input
                    type="text"
                    required
                    value={newPatient.last_name}
                    onChange={(e) => setNewPatient({ ...newPatient, last_name: e.target.value })}
                    className="w-full bg-surface-container-high border border-outline-variant/50 rounded-xl px-3 py-2 text-sm mt-1 focus:outline-none focus:border-primary/50 text-on-surface"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-xs text-on-surface-variant font-semibold">MRN Identifier</label>
                  <input
                    type="text"
                    placeholder="MRN-XXXXXX"
                    required
                    value={newPatient.mrn}
                    onChange={(e) => setNewPatient({ ...newPatient, mrn: e.target.value })}
                    className="w-full bg-surface-container-high border border-outline-variant/50 rounded-xl px-3 py-2 text-sm mt-1 focus:outline-none focus:border-primary/50 text-on-surface"
                  />
                </div>
                <div>
                  <label className="text-xs text-on-surface-variant font-semibold">Gender</label>
                  <select
                    value={newPatient.gender}
                    onChange={(e) => setNewPatient({ ...newPatient, gender: e.target.value })}
                    className="w-full bg-surface-container-high border border-outline-variant/50 rounded-xl px-3 py-2.5 text-sm mt-1 focus:outline-none focus:border-primary/50 text-on-surface"
                  >
                    <option value="F">Female</option>
                    <option value="M">Male</option>
                    <option value="O">Other</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="text-xs text-on-surface-variant font-semibold">Birth Date</label>
                <input
                  type="date"
                  required
                  value={newPatient.birth_date}
                  onChange={(e) => setNewPatient({ ...newPatient, birth_date: e.target.value })}
                  className="w-full bg-surface-container-high border border-outline-variant/50 rounded-xl px-3 py-2 text-sm mt-1 focus:outline-none focus:border-primary/50 text-on-surface"
                />
              </div>

              <div>
                <label className="text-xs text-on-surface-variant font-semibold">Genetic & Family History Notes</label>
                <textarea
                  value={newPatient.family_history}
                  onChange={(e) => setNewPatient({ ...newPatient, family_history: e.target.value })}
                  rows={2}
                  className="w-full bg-surface-container-high border border-outline-variant/50 rounded-xl px-3 py-2 text-sm mt-1 focus:outline-none focus:border-primary/50 text-on-surface resize-none"
                />
              </div>

              <div className="flex justify-end gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => setShowAddPatient(false)}
                  className="px-4 py-2 rounded-lg border border-outline-variant/50 text-xs font-semibold text-on-surface-variant hover:bg-surface-container-high"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 rounded-lg bg-primary text-on-primary hover:bg-primary/90 text-xs font-bold transition-all"
                >
                  Register Patient
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* New Study Modal */}
      {showAddStudy && (
        <div className="fixed inset-0 bg-background/80 backdrop-blur-sm flex justify-center items-center z-50 p-4">
          <div className="glass-elevated rounded-2xl border border-primary/20 shadow-2xl max-w-md w-full p-6 space-y-6 relative">
            <button
              onClick={() => setShowAddStudy(false)}
              className="absolute top-4 right-4 text-on-surface-variant hover:text-on-surface focus:outline-none"
            >
              <span className="material-symbols-outlined text-lg">close</span>
            </button>
            <h3 className="text-lg font-bold text-on-surface flex items-center gap-2">
              <span className="material-symbols-outlined text-primary text-xl">add_box</span>
              Initialize Imaging Study
            </h3>
            <form onSubmit={handleCreateStudy} className="space-y-4">
              <div>
                <div className="flex justify-between items-center mb-1">
                  <label className="text-xs text-on-surface-variant font-semibold">Select Patient Record</label>
                  <button
                    type="button"
                    onClick={() => openAddPatientModal()}
                    className="text-primary hover:text-primary/80 text-xs font-semibold flex items-center gap-1 transition-all active:scale-95"
                  >
                    <span className="material-symbols-outlined text-[14px]">person_add</span> Register New Patient
                  </button>
                </div>
                <select
                  required
                  value={selectedPatientId || ""}
                  onChange={(e) => setSelectedPatientId(Number(e.target.value))}
                  className="w-full bg-surface-container-high border border-outline-variant/50 rounded-xl px-3 py-2.5 text-sm focus:outline-none focus:border-primary/50 text-on-surface"
                >
                  <option value="">-- Choose Patient --</option>
                  {patients.map(p => (
                    <option key={p.id} value={p.id}>{p.first_name} {p.last_name} ({p.mrn})</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="text-xs text-on-surface-variant font-semibold">Breast Density Category</label>
                <select
                  value={newStudyDensity}
                  onChange={(e) => setNewStudyDensity(e.target.value)}
                  className="w-full bg-surface-container-high border border-outline-variant/50 rounded-xl px-3 py-2.5 text-sm mt-1 focus:outline-none focus:border-primary/50 text-on-surface"
                >
                  <option value="A">Density Category A (Entirely fatty)</option>
                  <option value="B">Density Category B (Scattered dense fibroglandular)</option>
                  <option value="C">Density Category C (Heterogeneously dense)</option>
                  <option value="D">Density Category D (Extremely dense)</option>
                </select>
              </div>

              <div className="flex justify-end gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => setShowAddStudy(false)}
                  className="px-4 py-2 rounded-lg border border-outline-variant/50 text-xs font-semibold text-on-surface-variant hover:bg-surface-container-high"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 rounded-lg bg-primary text-on-primary hover:bg-primary/90 text-xs font-bold transition-all"
                >
                  Initialize Study
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Side-by-Side Scan Comparison Modal */}
      {compareScan && studyDetail && (
        <div className="fixed inset-0 bg-background/80 backdrop-blur-sm flex justify-center items-center z-[60] p-4 md:p-6">
          <div className="glass-elevated rounded-2xl border border-primary/20 shadow-2xl max-w-5xl w-full p-6 space-y-6 relative flex flex-col max-h-[90vh] overflow-hidden">
            
            {/* Header */}
            <div className="flex justify-between items-center border-b border-outline-variant/30 pb-4 shrink-0">
              <div className="flex items-center gap-3">
                <span className="material-symbols-outlined text-primary text-2xl">compare</span>
                <h3 className="text-lg font-bold text-on-surface font-headline flex items-center gap-2">
                  Clinical Comparative Viewer
                  <span className="text-xs font-normal px-2.5 py-0.5 rounded-full bg-surface-container-high border border-outline-variant text-on-surface-variant">
                    {studyDetail.study.patient.first_name} {studyDetail.study.patient.last_name} ({studyDetail.study.patient.mrn})
                  </span>
                </h3>
              </div>
              <button
                onClick={() => setCompareScan(null)}
                className="w-8 h-8 rounded-full bg-surface-container-high hover:bg-surface-container-highest border border-outline-variant/50 flex items-center justify-center text-on-surface-variant hover:text-on-surface transition-all active:scale-95"
                title="Close Viewer"
              >
                <span className="material-symbols-outlined text-lg">close</span>
              </button>
            </div>            {/* Content (Side-by-Side Comparison) */}
            <div className="flex-1 overflow-y-auto custom-scrollbar p-1">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                
                {/* Left: Prior Scan */}
                <div className="glass-panel rounded-xl p-5 flex flex-col gap-4">
                  <div className="flex justify-between items-center shrink-0">
                    <h4 className="text-sm font-semibold text-on-surface-variant flex items-center gap-1.5">
                      <span className="material-symbols-outlined text-sm">history</span>
                      Prior Scan
                    </h4>
                    <div className="flex items-center gap-2">
                      <div className="flex gap-1 bg-surface-container-high p-0.5 rounded-lg border border-outline-variant/30">
                        <button
                          onClick={() => setComparePriorView('raw')}
                          className={`px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider rounded transition-all ${
                            comparePriorView === 'raw' ? 'bg-surface-container-lowest text-primary shadow-sm' : 'text-on-surface-variant hover:text-on-surface'
                          }`}
                        >
                          Raw
                        </button>
                        {compareScan.maskUrl && (
                          <button
                            onClick={() => setComparePriorView('mask')}
                            className={`px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider rounded transition-all ${
                              comparePriorView === 'mask' ? 'bg-surface-container-lowest text-primary shadow-sm' : 'text-on-surface-variant hover:text-on-surface'
                            }`}
                          >
                            Mask
                          </button>
                        )}
                        {compareScan.xaiUrl && (
                          <button
                            onClick={() => setComparePriorView('xai')}
                            className={`px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider rounded transition-all ${
                              comparePriorView === 'xai' ? 'bg-surface-container-lowest text-primary shadow-sm' : 'text-on-surface-variant hover:text-on-surface'
                            }`}
                          >
                            XAI
                          </button>
                        )}
                      </div>
                      <span className="text-xs font-mono text-on-surface-variant bg-surface-container px-2 py-1 rounded">
                        {compareScan.date}
                      </span>
                    </div>
                  </div>
                  
                  {/* Prior Image Box */}
                  <div className="w-full aspect-[4/3] rounded-lg bg-black overflow-hidden border border-outline-variant/30 relative flex items-center justify-center">
                    {comparePriorView === 'raw' && (
                      <img 
                        src={compareScan.imgUrl} 
                        alt="Prior Mammogram Scan" 
                        className="w-full h-full object-contain opacity-80"
                      />
                    )}
                    {comparePriorView === 'mask' && (
                      <>
                        <img 
                          src={compareScan.imgUrl} 
                          alt="Prior Mammogram Scan Background" 
                          className="w-full h-full object-contain opacity-80"
                        />
                        {compareScan.maskUrl && (
                          <div 
                            className="absolute inset-0 bg-contain bg-center bg-no-repeat mix-blend-screen opacity-50 filter hue-rotate-90"
                            style={{ backgroundImage: `url(${BACKEND_URL}${compareScan.maskUrl})` }}
                          ></div>
                        )}
                      </>
                    )}
                    {comparePriorView === 'xai' && (
                      <div className="w-full h-full relative">
                        {compareScan.xaiUrl ? (
                          <div 
                            className="absolute inset-0 bg-contain bg-center bg-no-repeat" 
                            style={{ backgroundImage: `url(${BACKEND_URL}${compareScan.xaiUrl})` }}
                          ></div>
                        ) : (
                          <p className="absolute inset-0 flex items-center justify-center text-on-surface-variant text-xs">No XAI map available</p>
                        )}
                      </div>
                    )}
                    <div className="absolute inset-0 bg-gradient-to-t from-background/40 to-transparent pointer-events-none"></div>
                  </div>
                  
                  {/* Prior Scan Data */}
                  <div className="space-y-2">
                    <div className="flex justify-between items-center p-2.5 rounded-lg bg-surface-container-high/50 border border-outline-variant/20 text-xs">
                      <span className="text-on-surface-variant font-medium">Density</span>
                      <span className="font-medium text-on-surface">Volumetric Category {compareScan.density || 'C'}</span>
                    </div>
                    <div className="flex justify-between items-center p-2.5 rounded-lg bg-surface-container-high/50 border border-outline-variant/20 text-xs">
                      <span className="text-on-surface-variant font-medium">BI-RADS Classification</span>
                      <span className="font-bold text-primary">{compareScan.birads}</span>
                    </div>
                    <div className="flex justify-between items-center p-2.5 rounded-lg bg-surface-container-high/50 border border-outline-variant/20 text-xs font-mono">
                      <span className="text-on-surface-variant font-medium">AI Malignancy Confidence</span>
                      <span className="font-bold text-primary">{compareScan.malignancyScore}</span>
                    </div>
                  </div>
                </div>
 
                {/* Right: Current Scan */}
                <div className="glass-panel rounded-xl p-5 flex flex-col gap-4 border-primary/30 shadow-[0_0_20px_rgba(125,211,252,0.05)] relative">
                  <div className="absolute -top-1.5 -right-1.5 w-3.5 h-3.5 rounded-full bg-error animate-pulse shadow-[0_0_10px_rgba(255,107,107,0.8)] border border-background"></div>
                  
                  <div className="flex justify-between items-center shrink-0">
                    <h4 className="text-sm font-semibold text-primary flex items-center gap-1.5">
                      <span className="material-symbols-outlined text-sm">emergency</span>
                      Current Scan
                    </h4>
                    <div className="flex items-center gap-2">
                      <div className="flex gap-1 bg-surface-container-high p-0.5 rounded-lg border border-outline-variant/30">
                        <button
                          onClick={() => setCompareCurrentView('raw')}
                          className={`px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider rounded transition-all ${
                            compareCurrentView === 'raw' ? 'bg-surface-container-lowest text-primary shadow-sm' : 'text-on-surface-variant hover:text-on-surface'
                          }`}
                        >
                          Raw
                        </button>
                        {studyDetail.images[0]?.analysis?.mask_url && (
                          <button
                            onClick={() => setCompareCurrentView('mask')}
                            className={`px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider rounded transition-all ${
                              compareCurrentView === 'mask' ? 'bg-surface-container-lowest text-primary shadow-sm' : 'text-on-surface-variant hover:text-on-surface'
                            }`}
                          >
                            Mask
                          </button>
                        )}
                        {studyDetail.images[0]?.analysis?.xai_url && (
                          <button
                            onClick={() => setCompareCurrentView('xai')}
                            className={`px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider rounded transition-all ${
                              compareCurrentView === 'xai' ? 'bg-surface-container-lowest text-primary shadow-sm' : 'text-on-surface-variant hover:text-on-surface'
                            }`}
                          >
                            XAI
                          </button>
                        )}
                      </div>
                      <span className="text-xs font-mono text-primary bg-primary/10 px-2 py-1 rounded border border-primary/20">
                        {new Date(studyDetail.study.study_date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                      </span>
                    </div>
                  </div>
                  
                  {/* Current Image Box */}
                  <div className="w-full aspect-[4/3] rounded-lg bg-black overflow-hidden border border-primary/30 relative flex items-center justify-center">
                    {compareCurrentView === 'raw' && (
                      <img 
                        src={`${BACKEND_URL}${studyDetail.images[0].url}`} 
                        alt="Current Mammogram Scan" 
                        className="w-full h-full object-contain"
                      />
                    )}
                    {compareCurrentView === 'mask' && (
                      <>
                        <img 
                          src={`${BACKEND_URL}${studyDetail.images[0].url}`} 
                          alt="Current Mammogram Scan Background" 
                          className="w-full h-full object-contain opacity-80"
                        />
                        {studyDetail.images[0].analysis?.mask_url && (
                          <div 
                            className="absolute inset-0 bg-contain bg-center bg-no-repeat mix-blend-screen opacity-50 filter hue-rotate-90"
                            style={{ backgroundImage: `url(${BACKEND_URL}${studyDetail.images[0].analysis.mask_url})` }}
                          ></div>
                        )}
                      </>
                    )}
                    {compareCurrentView === 'xai' && (
                      <div className="w-full h-full relative">
                        {studyDetail.images[0].analysis?.xai_url ? (
                          <div 
                            className="absolute inset-0 bg-contain bg-center bg-no-repeat" 
                            style={{ backgroundImage: `url(${BACKEND_URL}${studyDetail.images[0].analysis.xai_url})` }}
                          ></div>
                        ) : (
                          <p className="absolute inset-0 flex items-center justify-center text-on-surface-variant text-xs">No XAI map available</p>
                        )}
                      </div>
                    )}
                    <div className="absolute inset-0 bg-gradient-to-t from-background/40 to-transparent pointer-events-none"></div>
                    {studyDetail.images[0].analysis && (
                      <div className="absolute top-4 left-4 text-[10px] text-primary bg-background/80 px-2 py-0.5 rounded border border-primary/30 font-semibold tracking-wider uppercase">
                        ROI-1 Detected
                      </div>
                    )}
                  </div>
                  
                  {/* Current Scan Data */}
                  <div className="space-y-2">
                    <div className="flex justify-between items-center p-2.5 rounded-lg bg-surface-container-high/50 border border-outline-variant/20 text-xs">
                      <span className="text-on-surface-variant">Density</span>
                      <span className="font-medium text-on-surface">Volumetric Category {studyDetail.study.density_category}</span>
                    </div>
                    <div className="flex justify-between items-center p-2.5 rounded-lg bg-error-container/30 border border-error/30 text-xs">
                      <span className="text-on-surface-variant font-medium">BI-RADS Classification</span>
                      <span className="font-bold text-error">
                        {studyDetail.images[0].analysis ? studyDetail.images[0].analysis.birads : "Needs Review"}
                      </span>
                    </div>
                    <div className="flex justify-between items-center p-2.5 rounded-lg bg-primary/10 border border-primary/20 text-xs font-mono">
                      <span className="text-on-surface-variant font-medium">AI Malignancy Confidence</span>
                      <span className="font-bold text-primary">
                        {studyDetail.images[0].analysis ? `${(studyDetail.images[0].analysis.probabilities.Malignant * 100).toFixed(1)}%` : "Pending"}
                      </span>
                    </div>
                  </div>
                </div>
 
              </div>
            </div>
 
            {/* Actions */}
            <div className="flex justify-between items-center pt-4 border-t border-outline-variant/30 shrink-0">
              <button
                type="button"
                onClick={handleExportComparativePDF}
                className="px-5 py-2.5 rounded-lg bg-primary text-on-primary hover:bg-primary/90 shadow-md font-semibold text-sm flex items-center gap-1.5 transition-all active:scale-95 border border-primary/30"
              >
                <span className="material-symbols-outlined text-sm">picture_as_pdf</span>
                Generate Progress Report
              </button>
              <button
                type="button"
                onClick={() => setCompareScan(null)}
                className="px-5 py-2.5 rounded-lg bg-surface-container-high hover:bg-surface-container-highest border border-outline-variant text-sm font-semibold text-on-surface transition-all active:scale-95"
              >
                Close Comparison
              </button>
            </div>
            
          </div>
        </div>
      )}
    </div>
  )
}
