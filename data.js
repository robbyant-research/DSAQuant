window.SUPPLEMENT_DATA = {
  "meta": {
    "title": "DSAQuant: Denoising-Stage-Aligned Quantization-Aware Training for Video Generation",
    "subtitle": "Project page and qualitative video comparisons.",
    "authors": [
      {
        "name": "Shuaiting Li",
        "affiliations": [1, 2],
        "url": "https://list0830.github.io/"
      },
      {
        "name": "Zelin Gao",
        "affiliations": [1, 4],
        "url": "https://zelingao98.github.io/"
      },
      {
        "name": "Haibin Shen",
        "affiliations": [2],
        "url": "https://person.zju.edu.cn/en/7fj86kj"
      },
      {
        "name": "Yujun Shen",
        "affiliations": [1],
        "url": "https://shenyujun.github.io/index.html"
      },
      {
        "name": "Haotong Qin",
        "affiliations": [3],
        "corresponding": true,
        "url": "https://htqin.github.io/"
      },
      {
        "name": "Yinghao Xu",
        "affiliations": [4, 1],
        "corresponding": true,
        "url": "https://justimyhxu.github.io/"
      }
    ],
    "affiliations": [
      {
        "id": 1,
        "name": "Robby Ant"
      },
      {
        "id": 2,
        "name": "ZJU"
      },
      {
        "id": 3,
        "name": "PolyU"
      },
      {
        "id": 4,
        "name": "HKUST"
      }
    ]
  },
  "sections": [
    {
      "id": "system-level-comparison",
      "title": "Comparison with previous methods",
      "groups": [
        {
          "id": "wan-13b",
          "title": "WAN 1.3B",
          "samples": [
            {
              "id": "elephant-walk",
              "prompt": "CG animation digital art, an African elephant taking a peaceful walk through a lush green forest at dawn. The elephant has a grayish-brown coat with soft folds and wrinkles, and gentle eyes. It walks slowly, with its trunk raised slightly, sniffing the air. The forest is filled with tall trees and vibrant greenery, with colorful flowers blooming. Birds fly overhead, chirping melodiously. The sky is a beautiful orange hue as the sun rises. The elephant moves gracefully, with each step careful and deliberate. The background features intricate foliage and subtle shadows. Soft lighting creates a warm and serene atmosphere. Low-angle shot from behind, focusing on the elephant's tranquil expression and movement.",
              "comparisons": [
                {
                  "id": "elephant-walk-4bit",
                  "title": "4-bit",
                  "methods": [
                    {
                      "id": "fp",
                      "label": "FP"
                    },
                    {
                      "id": "qdm",
                      "label": "QDM"
                    },
                    {
                      "id": "qvgen",
                      "label": "QVGen"
                    },
                    {
                      "id": "ours",
                      "label": "Ours"
                    }
                  ],
                  "videos": {
                    "fp": "videos/system_level/elephant_fp.mp4",
                    "qdm": "videos/system_level/elephant_qdm.mp4",
                    "qvgen": "videos/system_level/elephant_qvgen.mp4",
                    "ours": "videos/system_level/elephant_ours.mp4"
                  }
                },
                {
                  "id": "elephant-walk-3bit",
                  "title": "3-bit",
                  "methods": [
                    {
                      "id": "fp",
                      "label": "FP"
                    },
                    {
                      "id": "qdm",
                      "label": "QDM"
                    },
                    {
                      "id": "qvgen",
                      "label": "QVGen"
                    },
                    {
                      "id": "ours",
                      "label": "Ours"
                    }
                  ],
                  "videos": {
                    "fp": "videos/system_level/elephant_fp.mp4",
                    "qdm": "videos/system_level/elephant_3bit_qdm.mp4",
                    "qvgen": "videos/system_level/elephant_3bit_qvgen.mp4",
                    "ours": "videos/system_level/elephant_3bit_ours.mp4"
                  }
                }
              ]
            },
            {
              "id": "jellyfish-ocean",
              "prompt": "CG animation digital art, a majestic jellyfish floating gracefully through the oceanic depths. The jellyfish has intricate patterns on its translucent body, with vibrant hues of blue, green, and purple. Its bioluminescent tentacles emit a soft, mesmerizing glow, creating an ethereal underwater landscape. The tentacles sway gently as the jellyfish glides, illuminating the surrounding water with a captivating light show. The ocean background is filled with schools of colorful fish and drifting coral reefs. The jellyfish is surrounded by a serene, tranquil atmosphere. Soft, ambient ocean sounds play in the background. Low-angle, slow-motion shot focusing on the jellyfish and its glowing tentacles.",
              "comparisons": [
                {
                  "id": "jellyfish-ocean-4bit",
                  "title": "4-bit",
                  "methods": [
                    {
                      "id": "fp",
                      "label": "FP"
                    },
                    {
                      "id": "qdm",
                      "label": "QDM"
                    },
                    {
                      "id": "qvgen",
                      "label": "QVGen"
                    },
                    {
                      "id": "ours",
                      "label": "Ours"
                    }
                  ],
                  "videos": {
                    "fp": "videos/system_level/jellyfish_fp.mp4",
                    "qdm": "videos/system_level/jellyfish_qdm.mp4",
                    "qvgen": "videos/system_level/jellyfish_qvgen.mp4",
                    "ours": "videos/system_level/jellyfish_ours.mp4"
                  }
                },
                {
                  "id": "jellyfish-ocean-3bit",
                  "title": "3-bit",
                  "methods": [
                    {
                      "id": "fp",
                      "label": "FP"
                    },
                    {
                      "id": "qdm",
                      "label": "QDM"
                    },
                    {
                      "id": "qvgen",
                      "label": "QVGen"
                    },
                    {
                      "id": "ours",
                      "label": "Ours"
                    }
                  ],
                  "videos": {
                    "fp": "videos/system_level/jellyfish_fp.mp4",
                    "qdm": "videos/system_level/jellyfish_3bit_qdm.mp4",
                    "qvgen": "videos/system_level/jellyfish_3bit_qvgen.mp4",
                    "ours": "videos/system_level/jellyfish_3bit_ours.mp4"
                  }
                }
              ]
            }
          ]
        },
        {
          "id": "wan-21-14b",
          "title": "WAN 2.1 14B",
          "samples": [
            {
              "id": "makeup-morning-wan21-14b",
              "prompt": "CG animation concept art, a young woman with natural beauty applying makeup in the morning. She is wearing a simple white blouse and black pencil skirt. Her hair is styled in loose waves, framing her face. She is sitting at a vanity table with soft lighting illuminating her work. She is using a compact mirror to apply foundation, concealer, and blush. Her fingers move gracefully as she blends and applies products. She pauses occasionally to check her reflection and adjust her technique. The background features a minimalist room with a few scattered items and a vintage vanity set. Soft, gentle brush strokes and subtle motions. Low-angle shot from above, focusing on her hands and facial expressions.",
              "comparisons": [
                {
                  "id": "makeup-morning-wan21-14b-4bit",
                  "title": "4-bit",
                  "methods": [
                    {
                      "id": "fp",
                      "label": "FP"
                    },
                    {
                      "id": "qvgen",
                      "label": "QVGen"
                    },
                    {
                      "id": "ours",
                      "label": "Ours"
                    }
                  ],
                  "videos": {
                    "fp": "videos/system_level/wan21_14b_makeup_fp.mp4",
                    "qvgen": "videos/system_level/wan21_14b_makeup_w4_qvgen.mp4",
                    "ours": "videos/system_level/wan21_14b_makeup_w4_ours.mp4"
                  }
                },
                {
                  "id": "makeup-morning-wan21-14b-3bit",
                  "title": "3-bit",
                  "methods": [
                    {
                      "id": "fp",
                      "label": "FP"
                    },
                    {
                      "id": "qvgen",
                      "label": "QVGen"
                    },
                    {
                      "id": "ours",
                      "label": "Ours"
                    }
                  ],
                  "videos": {
                    "fp": "videos/system_level/wan21_14b_makeup_fp.mp4",
                    "qvgen": "videos/system_level/wan21_14b_makeup_w3_qvgen.mp4",
                    "ours": "videos/system_level/wan21_14b_makeup_w3_ours.mp4"
                  }
                }
              ]
            },
            {
              "id": "motorcycle-accelerating-wan21-14b",
              "prompt": "A thrilling motorcycle speeding down a winding mountain road at night. The sleek black motorcycle with bright red accents accelerates rapidly, leaving behind a trail of dust. The rider, a muscular man with short cropped hair, wears a black leather jacket and jeans. His helmet is off, revealing a determined and focused expression. He grips the handlebars tightly, leaning slightly into the turn. The motorcycle's engine roars as it gains momentum, reflecting the intense speed and power. The background is a dimly lit mountain landscape, with flickering streetlights and twinkling stars. The scene captures the adrenaline rush and raw energy of the motorcycle's acceleration. Nighttime low-angle shot from the side.",
              "comparisons": [
                {
                  "id": "motorcycle-accelerating-wan21-14b-4bit",
                  "title": "4-bit",
                  "methods": [
                    {
                      "id": "fp",
                      "label": "FP"
                    },
                    {
                      "id": "qvgen",
                      "label": "QVGen"
                    },
                    {
                      "id": "ours",
                      "label": "Ours"
                    }
                  ],
                  "videos": {
                    "fp": "videos/system_level/wan21_14b_motorcycle_fp.mp4",
                    "qvgen": "videos/system_level/wan21_14b_motorcycle_w4_qvgen.mp4",
                    "ours": "videos/system_level/wan21_14b_motorcycle_w4_ours.mp4"
                  }
                },
                {
                  "id": "motorcycle-accelerating-wan21-14b-3bit",
                  "title": "3-bit",
                  "methods": [
                    {
                      "id": "fp",
                      "label": "FP"
                    },
                    {
                      "id": "qvgen",
                      "label": "QVGen"
                    },
                    {
                      "id": "ours",
                      "label": "Ours"
                    }
                  ],
                  "videos": {
                    "fp": "videos/system_level/wan21_14b_motorcycle_fp.mp4",
                    "qvgen": "videos/system_level/wan21_14b_motorcycle_w3_qvgen.mp4",
                    "ours": "videos/system_level/wan21_14b_motorcycle_w3_ours.mp4"
                  }
                }
              ]
            }
          ]
        },
        {
          "id": "wan-22-5b",
          "title": "WAN 2.2 5B",
          "samples": [
            {
              "id": "bicycle-wan22-5b",
              "prompt": "CG animation digital art, a sleek racing bicycle speeding down a winding mountain road. The bike is a deep metallic silver color with intricate patterns etched along its frame. It accelerates rapidly, the rider gripping the handlebars tightly with focused determination. The rider has short, wavy brown hair and piercing blue eyes, wearing a tight-fitting black helmet and a snugly fitting silver jersey. They lean forward slightly, their arms pumping hard to maintain momentum. The road is rugged, covered in loose gravel and rocks, with trees lining the sides. The sun sets behind them, casting dramatic shadows. The background features a sunset sky with wispy clouds and hints of orange and pink hues. The bike's wheels spin furiously as it gains speed, creating a whirlwind of dust and debris. The scene captures the adrenaline rush of a thrilling bicycle race. Close-up, low-angle view.",
              "comparisons": [
                {
                  "id": "bicycle-wan22-5b-4bit",
                  "title": "4-bit",
                  "methods": [
                    {
                      "id": "fp",
                      "label": "FP"
                    },
                    {
                      "id": "qvgen",
                      "label": "QVGen"
                    },
                    {
                      "id": "ours",
                      "label": "Ours"
                    }
                  ],
                  "videos": {
                    "fp": "videos/system_level/wan22_5b_bicycle_fp.mp4",
                    "qvgen": "videos/system_level/wan22_5b_bicycle_w4_qvgen.mp4",
                    "ours": "videos/system_level/wan22_5b_bicycle_w4_ours.mp4"
                  }
                },
                {
                  "id": "bicycle-wan22-5b-3bit",
                  "title": "3-bit",
                  "methods": [
                    {
                      "id": "fp",
                      "label": "FP"
                    },
                    {
                      "id": "qvgen",
                      "label": "QVGen"
                    },
                    {
                      "id": "ours",
                      "label": "Ours"
                    }
                  ],
                  "videos": {
                    "fp": "videos/system_level/wan22_5b_bicycle_fp.mp4",
                    "qvgen": "videos/system_level/wan22_5b_bicycle_w3_qvgen.mp4",
                    "ours": "videos/system_level/wan22_5b_bicycle_w3_ours.mp4"
                  }
                }
              ]
            },
            {
              "id": "cat-wan22-5b",
              "prompt": "A playful feline sprinting joyfully across a lush green meadow dotted with wildflowers. The cat has sleek fur, expressive green eyes, and a fluffy tail that wags excitedly as it bounds forward. The meadow stretches out behind it, with vibrant sunflowers and buttercups swaying gently in the breeze. The sky above is a bright azure, filled with fluffy white clouds. The cat's joyful run is captured from a dynamic low-angle perspective, showcasing its agility and boundless energy. The scene is bathed in warm golden light, enhancing the cat's lively demeanor. Grass and petals trail behind the cat as it dashes towards the horizon. The background features a serene rural landscape, with small cottages and winding country roads visible in the distance. The overall composition is energetic and full of life, perfectly capturing the essence of a cat running happily.",
              "comparisons": [
                {
                  "id": "cat-wan22-5b-4bit",
                  "title": "4-bit",
                  "methods": [
                    {
                      "id": "fp",
                      "label": "FP"
                    },
                    {
                      "id": "qvgen",
                      "label": "QVGen"
                    },
                    {
                      "id": "ours",
                      "label": "Ours"
                    }
                  ],
                  "videos": {
                    "fp": "videos/system_level/wan22_5b_cat_fp.mp4",
                    "qvgen": "videos/system_level/wan22_5b_cat_w4_qvgen.mp4",
                    "ours": "videos/system_level/wan22_5b_cat_w4_ours.mp4"
                  }
                },
                {
                  "id": "cat-wan22-5b-3bit",
                  "title": "3-bit",
                  "methods": [
                    {
                      "id": "fp",
                      "label": "FP"
                    },
                    {
                      "id": "qvgen",
                      "label": "QVGen"
                    },
                    {
                      "id": "ours",
                      "label": "Ours"
                    }
                  ],
                  "videos": {
                    "fp": "videos/system_level/wan22_5b_cat_fp.mp4",
                    "qvgen": "videos/system_level/wan22_5b_cat_w3_qvgen.mp4",
                    "ours": "videos/system_level/wan22_5b_cat_w3_ours.mp4"
                  }
                }
              ]
            }
          ]
        }
      ]
    },
    {
      "id": "ablation-study",
      "title": "Ablation study",
      "groups": [
        {
          "id": "wan-13b-ablation",
          "title": "WAN 1.3B",
          "samples": [
            {
              "id": "a_drone_flying_over_a_snowy_forest_4",
              "prompt": "a drone flying over a snowy forest.",
              "comparisons": [
                {
                  "id": "a_drone_flying_over_a_snowy_forest_4-ablation",
                  "title": "",
                  "methods": [
                    {
                      "id": "fp",
                      "label": "FP"
                    },
                    {
                      "id": "base",
                      "label": "Base"
                    },
                    {
                      "id": "supervision",
                      "label": "+ Denoising-stage-oriented supervision"
                    },
                    {
                      "id": "gated",
                      "label": "+ Denoising-stage Gated Guidance"
                    }
                  ],
                  "videos": {
                    "fp": "videos/ablation/a_drone_flying_over_a_snowy_forest_4__fp.mp4",
                    "base": "videos/ablation/a_drone_flying_over_a_snowy_forest_4__base.mp4",
                    "supervision": "videos/ablation/a_drone_flying_over_a_snowy_forest_4__supervision.mp4",
                    "gated": "videos/ablation/a_drone_flying_over_a_snowy_forest_4__gated.mp4"
                  }
                }
              ]
            },
            {
              "id": "turtle_swimming_in_ocean_4",
              "prompt": "Turtle swimming in ocean.",
              "comparisons": [
                {
                  "id": "turtle_swimming_in_ocean_4-ablation",
                  "title": "",
                  "methods": [
                    {
                      "id": "fp",
                      "label": "FP"
                    },
                    {
                      "id": "base",
                      "label": "Base"
                    },
                    {
                      "id": "supervision",
                      "label": "+ Denoising-stage-oriented supervision"
                    },
                    {
                      "id": "gated",
                      "label": "+ Denoising-stage Gated Guidance"
                    }
                  ],
                  "videos": {
                    "fp": "videos/ablation/turtle_swimming_in_ocean_4__fp.mp4",
                    "base": "videos/ablation/turtle_swimming_in_ocean_4__base.mp4",
                    "supervision": "videos/ablation/turtle_swimming_in_ocean_4__supervision.mp4",
                    "gated": "videos/ablation/turtle_swimming_in_ocean_4__gated.mp4"
                  }
                }
              ]
            },
            {
              "id": "an_astronaut_is_riding_a_horse_in_the_space_in_a_photorealistic_style_4",
              "prompt": "An astronaut is riding a horse in the space in a photorealistic style.",
              "comparisons": [
                {
                  "id": "an_astronaut_is_riding_a_horse_in_the_space_in_a_photorealistic_style_4-ablation",
                  "title": "",
                  "methods": [
                    {
                      "id": "fp",
                      "label": "FP"
                    },
                    {
                      "id": "base",
                      "label": "Base"
                    },
                    {
                      "id": "supervision",
                      "label": "+ Denoising-stage-oriented supervision"
                    },
                    {
                      "id": "gated",
                      "label": "+ Denoising-stage Gated Guidance"
                    }
                  ],
                  "videos": {
                    "fp": "videos/ablation/an_astronaut_is_riding_a_horse_in_the_space_in_a_photorealistic_style_4__fp.mp4",
                    "base": "videos/ablation/an_astronaut_is_riding_a_horse_in_the_space_in_a_photorealistic_style_4__base.mp4",
                    "supervision": "videos/ablation/an_astronaut_is_riding_a_horse_in_the_space_in_a_photorealistic_style_4__supervision.mp4",
                    "gated": "videos/ablation/an_astronaut_is_riding_a_horse_in_the_space_in_a_photorealistic_style_4__gated.mp4"
                  }
                }
              ]
            }
          ]
        }
      ]
    }
  ]
};
