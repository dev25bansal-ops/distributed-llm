type ToastType = "success" | "error" | "info" | "warning";

interface Toast {
  id: number;
  message: string;
  type: ToastType;
}

let toasts: Toast[] = [];
let listeners: Array<(toasts: Toast[]) => void> = [];
let nextId = 0;

function notify() {
  for (const fn of listeners) {
    fn([...toasts]);
  }
}

function addToast(message: string, type: ToastType, durationMs = 4000) {
  const id = nextId++;
  toasts = [...toasts, { id, message, type }];
  notify();

  setTimeout(() => {
    removeToast(id);
  }, durationMs);
}

function removeToast(id: number) {
  toasts = toasts.filter((t) => t.id !== id);
  notify();
}

export const toastStore = {
  subscribe(fn: (toasts: Toast[]) => void) {
    listeners.push(fn);
    fn(toasts);
    return () => {
      listeners = listeners.filter((l) => l !== fn);
    };
  },
  success(message: string) {
    addToast(message, "success");
  },
  error(message: string) {
    addToast(message, "error", 6000);
  },
  info(message: string) {
    addToast(message, "info");
  },
  warning(message: string) {
    addToast(message, "warning", 5000);
  },
  dismiss(id: number) {
    removeToast(id);
  },
};
