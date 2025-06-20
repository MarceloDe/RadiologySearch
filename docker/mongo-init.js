# MongoDB initialization script
# Creates initial database structure and indexes

print("🏥 Initializing Radiology AI Database...");

// Switch to the radiology database
db = db.getSiblingDB('radiology_ai_langchain');

// Create collections with validation
db.createCollection("prompts", {
   validator: {
      $jsonSchema: {
         bsonType: "object",
         required: ["template_id", "name", "version", "template_text", "model_type"],
         properties: {
            template_id: {
               bsonType: "string",
               description: "Unique template identifier"
            },
            name: {
               bsonType: "string",
               description: "Human-readable template name"
            },
            version: {
               bsonType: "string",
               description: "Template version"
            },
            template_text: {
               bsonType: "string",
               description: "Actual prompt template"
            },
            model_type: {
               bsonType: "string",
               enum: ["claude", "mistral", "deepseek"],
               description: "Target model type"
            },
            performance_metrics: {
               bsonType: "object",
               description: "Performance tracking data"
            }
         }
      }
   }
});

db.createCollection("cases", {
   validator: {
      $jsonSchema: {
         bsonType: "object",
         required: ["case_id", "patient_age", "patient_sex", "clinical_history"],
         properties: {
            case_id: {
               bsonType: "string",
               description: "Unique case identifier"
            },
            patient_age: {
               bsonType: "int",
               minimum: 0,
               maximum: 120,
               description: "Patient age in years"
            },
            patient_sex: {
               bsonType: "string",
               enum: ["Male", "Female", "Other"],
               description: "Patient biological sex"
            },
            clinical_history: {
               bsonType: "string",
               description: "Clinical history and symptoms"
            }
         }
      }
   }
});

db.createCollection("analysis_results", {
   validator: {
      $jsonSchema: {
         bsonType: "object",
         required: ["case_id", "analysis_timestamp"],
         properties: {
            case_id: {
               bsonType: "string",
               description: "Reference to case"
            },
            analysis_timestamp: {
               bsonType: "date",
               description: "When analysis was performed"
            }
         }
      }
   }
});

db.createCollection("literature_cache", {
   validator: {
      $jsonSchema: {
         bsonType: "object",
         required: ["search_query", "results", "cached_at"],
         properties: {
            search_query: {
               bsonType: "string",
               description: "Original search query"
            },
            results: {
               bsonType: "array",
               description: "Cached search results"
            },
            cached_at: {
               bsonType: "date",
               description: "Cache timestamp"
            }
         }
      }
   }
});

// Create indexes for performance
print("📊 Creating database indexes...");

// Prompts indexes
db.prompts.createIndex({ "template_id": 1, "version": -1 }, { unique: true });
db.prompts.createIndex({ "model_type": 1 });
db.prompts.createIndex({ "created_at": -1 });

// Cases indexes
db.cases.createIndex({ "case_id": 1 }, { unique: true });
db.cases.createIndex({ "patient_age": 1, "patient_sex": 1 });
db.cases.createIndex({ "imaging_modality": 1, "anatomical_region": 1 });
db.cases.createIndex({ "created_at": -1 });

// Analysis results indexes
db.analysis_results.createIndex({ "case_id": 1 });
db.analysis_results.createIndex({ "analysis_timestamp": -1 });
db.analysis_results.createIndex({ "diagnosis_result.primary_diagnosis.diagnosis": 1 });

// Literature cache indexes
db.literature_cache.createIndex({ "search_query": 1 }, { unique: true });
db.literature_cache.createIndex({ "cached_at": 1 }, { expireAfterSeconds: 86400 }); // 24 hours TTL

// Create user for application
db.createUser({
   user: "radiology_app",
   pwd: "radiology_secure_2024",
   roles: [
      {
         role: "readWrite",
         db: "radiology_ai_langchain"
      }
   ]
});

print("✅ Database initialization completed!");
print("📋 Collections created: prompts, cases, analysis_results, literature_cache");
print("🔍 Indexes created for optimal performance");
print("👤 Application user created: radiology_app");

