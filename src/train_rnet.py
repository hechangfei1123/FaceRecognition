import net
import train

if __name__ == '__main__':
    ne = net.RNet()
    ne.train()
    trainer = train.Trainer(ne, 'E:/save_path/24/param/',r"E:\save_path\24")
    trainer.train()