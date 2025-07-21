"""Point d'entrée et tests pour l'agent CV"""

import os
from agent import CVAgent

def test_cv_processing():
    """Test du traitement d'un CV"""
    print("📄 Test 1: Traitement d'un CV")
    cv_path = r"C:\Users\ASUS\Desktop\stage\cv\BedouiDalenda.pdf"
    
    if os.path.exists(cv_path):
        agent = CVAgent()
        result = agent.process_cv(cv_path)
        print("Résultat:", result["messages"][-1].content)
        return True
    else:
        print("❌ Fichier CV non trouvé")
        return False

def test_cv_search():
    """Test de recherche de CV"""
    print("\n🔍 Test 2: Recherche de compétences")
    agent = CVAgent()
    result = agent.search_cvs("Python développeur")
    print("Résultat:", result["messages"][-1].content)

def test_general_chat():
    """Test du chat général"""
    print("\n💬 Test 3: Chat général")
    agent = CVAgent()
    response = agent.chat("Que peux-tu faire avec les CV?")
    print("Réponse:", response)

def interactive_mode():
    """Interface interactive"""
    print("\n🔄 Interface interactive (tapez 'quit' pour quitter)")
    agent = CVAgent()
    
    while True:
        user_input = input("\nVous: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        # Traitement spécial pour les chemins de fichiers CV
        if user_input.startswith("cv:"):
            cv_path = user_input[3:].strip()
            if os.path.exists(cv_path):
                result = agent.process_cv(cv_path)
                print(f"Agent: {result['messages'][-1]['content']}")
            else:
                print("Agent: ❌ Fichier CV non trouvé")
        else:
            response = agent.chat(user_input)
            print(f"Agent: {response}")

def main():
    """Fonction principale"""
    print("🤖 Initialisation de l'agent CV...")
    
    # Exécuter les tests
    cv_processed = test_cv_processing()
    
    if cv_processed:
        test_cv_search()
    
    test_general_chat()
    
    # Interface interactive
    interactive_mode()
    
    print("\n👋 Au revoir!")

if __name__ == "__main__":
    main()