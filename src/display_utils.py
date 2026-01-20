def display_yolo_evaluation(stats):
    print("\n" + "="*70)
    print("  ÉVALUATION - PERFORMANCE YOLO")
    print("="*70)
    
    # Statistiques globales
    print(f"\n Statistiques globales:")
    print(f"  • Total d'images traitées       : {stats['total_images']}")
    print(f"  • Images avec détection         : {stats['images_with_detection']}")
    print(f"  • Images sans détection         : {stats['images_without_detection']}")
    print(f"  • Taux de détection             : {stats.get('detection_rate_%', 0):.2f}%")
    
    # Détections de personnes
    print(f"\n Détections de personnes:")
    print(f"  • Total de personnes détectées  : {stats['total_persons_detected']}")
    print(f"  • Total de crops sauvegardés    : {stats['total_crops_saved']}")
    print(f"  • Images multi-personnes        : {stats['multiple_persons_images']}")
    print(f"  • Moyenne personnes/image       : {stats.get('avg_persons_per_image', 0):.2f}")
    
    # Métriques de confiance
    if stats['confidence_scores']:
        avg_conf = sum(stats['confidence_scores']) / len(stats['confidence_scores'])
        min_conf = min(stats['confidence_scores'])
        max_conf = max(stats['confidence_scores'])
        
        print(f"\n Scores de confiance:")
        print(f"  • Confiance moyenne             : {avg_conf:.3f}")
        print(f"  • Confiance minimale            : {min_conf:.3f}")
        print(f"  • Confiance maximale            : {max_conf:.3f}")
        print(f"  • Détections low-conf (<0.5)    : {stats['low_confidence_count']}")
    
    # Métriques de taille
    if stats['box_sizes']:
        avg_size = sum(stats['box_sizes']) / len(stats['box_sizes'])
        min_size = min(stats['box_sizes'])
        max_size = max(stats['box_sizes'])
        
        print(f"\n Qualité des bounding boxes:")
        print(f"  • Taille moyenne (pixels²)      : {avg_size:.0f}")
        print(f"  • Taille minimale               : {min_size}")
        print(f"  • Taille maximale               : {max_size}")
        print(f"  • Boxes anormales (ratio)       : {stats['abnormal_boxes_count']}")
    
    print("\n" + "="*70)
    print(f" Fichiers générés dans 'working/' : {stats['total_crops_saved']}")
    print(f" Métadonnées sauvegardées dans 'metadata.json'")
    print("="*70)
    print()


